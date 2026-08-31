"""Turning detector output into something a KYC analyst can act on in 20 seconds.

Two products here:
  * `classify_attack` - names the *pattern*, so the ops team sees fraud waves
    rather than a list of unrelated rejections;
  * `narrate` - a plain-English paragraph with the evidence and confidence for
    each claim. Deterministic templating, no LLM in the request path: it must
    never hallucinate a reason the detectors did not actually produce.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from .config import DETECTOR_LABELS
from .schemas import DetectorOutput

#: Ordered most-specific-first; the first pattern whose predicate fires wins.
ATTACK_PATTERNS: List[Tuple[str, str]] = [
    ("reused_kyc_kit", "Reused KYC kit"),
    ("onboarding_ring", "Onboarding ring (one face, many identities)"),
    ("automated_submission", "Automated submission (bot-filled form)"),
    ("assisted_submission", "Form filled by someone other than the applicant"),
    ("reused_id_selfie", "Selfie copied from ID photo"),
    ("impersonation", "Impersonation (selfie ≠ ID holder)"),
    ("synthetic_identity", "Fully synthetic identity"),
    ("face_swap_liveness", "Face-swapped liveness video"),
    ("generated_selfie", "AI-generated selfie"),
    ("tampered_document", "Tampered ID document"),
    ("invalid_document", "Structurally invalid ID number"),
    ("recaptured_screen", "Screen re-capture / photo of a photo"),
    ("stale_or_edited_media", "Stale or edited media"),
    ("clean", "No attack pattern detected"),
]
PATTERN_LABELS = dict(ATTACK_PATTERNS)


def _sig(b: Dict[str, DetectorOutput], det: str, key: str, default=None):
    d = b.get(det)
    if d is None or d.status != "ok":
        return default
    return d.signals.get(key, default)


def _score(b: Dict[str, DetectorOutput], det: str) -> float:
    d = b.get(det)
    return float(d.score) if d is not None and d.status == "ok" else 0.0


def classify_attack(breakdown: Dict[str, DetectorOutput], linkage: Optional[Dict] = None) -> str:
    linkage = linkage or {}
    selfie = _score(breakdown, "selfie_deepfake")
    video = _score(breakdown, "liveness_video")
    doc = _score(breakdown, "id_forensics")
    meta = _score(breakdown, "metadata_exif")

    # Only an exact pixel match earns the kit label. A perceptual near-match is
    # what two documents sharing a template look like, and naming that a reused
    # kit would put a fraud pattern on a genuine merchant.
    if any(m.get("match") == "exact" for m in linkage.get("asset_links", []) or []):
        return "reused_kyc_kit"
    if any(m.get("name_differs") for m in linkage.get("face_links", [])):
        return "onboarding_ring"

    # Behavioral evidence is checked early because it names something the
    # artifact detectors cannot: not what was uploaded, but who - or what - was
    # at the keyboard. A categorical hit outranks any image finding, since a
    # browser that declares itself automated has settled the question of whether
    # a human filled this form regardless of how clean the selfie looks.
    behav_hits = {h["rule"] for h in (_sig(breakdown, "behavioral", "rule_hits", []) or [])}
    if behav_hits & {"environment:automation", "keystroke:below_human_floor",
                     "keystroke:uniform_dwell", "keystroke:uniform_flight",
                     "pointer:uniform_sampling", "timing:hardcoded_sleep"}:
        return "automated_submission"
    # A human typed it, but the evidence says they were working from a script of
    # their own: the identity number pasted from a file, no hesitation anywhere.
    # That is an agent onboarding merchants in bulk, not necessarily a bot.
    if "form:pasted_identity" in behav_hits and _score(breakdown, "behavioral") >= 0.5:
        return "assisted_submission"

    direction = _sig(breakdown, "face_match", "direction")
    meta_hits = [h["rule"] for h in (_sig(breakdown, "metadata_exif", "rule_hits", []) or [])]
    upscaled_selfie = any(h.startswith("selfie:upscaled") for h in meta_hits)
    if direction == "duplicate" or (upscaled_selfie and direction == "match"):
        # Either the embeddings are implausibly identical, or the "selfie" is an
        # enlargement of something card-sized while still matching the ID holder.
        return "reused_id_selfie"
    if direction == "mismatch":
        return "impersonation"

    id_photo_fake = float(_sig(breakdown, "id_forensics", "id_photo_p_fake", 0.0) or 0.0)
    if selfie > 0.65 and id_photo_fake > 0.65:
        return "synthetic_identity"

    video_sub = _sig(breakdown, "liveness_video", "subscores", {}) or {}
    if video > 0.6 and float(video_sub.get("temporal", 0.0)) > 0.5:
        return "face_swap_liveness"
    if selfie > 0.65:
        return "generated_selfie"

    doc_sub = _sig(breakdown, "id_forensics", "subscores", {}) or {}
    if doc > 0.5 and float(doc_sub.get("tamper", 0.0)) > 0.45:
        return "tampered_document"
    if doc > 0.5 and float(doc_sub.get("fields", 0.0)) > 0.5:
        return "invalid_document"

    hits = meta_hits
    if any("screenshot" in h for h in hits):
        return "recaptured_screen"
    if meta > 0.5:
        return "stale_or_edited_media"
    return "clean"


def rank_reasons(breakdown: Dict[str, DetectorOutput], extra: Optional[List[Tuple[float, str]]] = None, top_n: int = 3) -> List[str]:
    """Rank every reason by how much evidential weight sits behind it."""
    scored: List[Tuple[float, str]] = list(extra or [])
    for name, d in breakdown.items():
        if d.status != "ok":
            continue
        weight = d.score * max(d.confidence, 0.1)
        for i, r in enumerate(d.reasons):
            # First reason from a detector carries its full weight; later ones taper.
            scored.append((weight * (1.0 - 0.15 * i), r))
    scored.sort(key=lambda t: t[0], reverse=True)

    out: List[str] = []
    seen = set()
    for _, r in scored:
        key = r[:60].lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
        if len(out) >= top_n:
            break
    return out


def _confidence_phrase(d: DetectorOutput) -> str:
    return f"{d.score:.0%} confidence"


def narrate(
    verdict: str,
    final_score: float,
    breakdown: Dict[str, DetectorOutput],
    attack_pattern: str,
    generator: Optional[str] = None,
    linkage: Optional[Dict] = None,
    abstained: bool = False,
) -> str:
    """A paragraph a non-technical reviewer can act on, built only from real signals."""
    verb = {"REJECT": "rejected", "REVIEW": "sent to human review", "PASS": "approved"}[verdict]
    lead = f"This submission was {verb} with an overall risk score of {final_score:.2f}."

    if verdict == "PASS":
        clean_bits = []
        for name in ("selfie_deepfake", "id_forensics", "face_match", "liveness_video"):
            d = breakdown.get(name)
            if d is not None and d.status == "ok" and d.score < 0.35:
                clean_bits.append(DETECTOR_LABELS[name].lower())
        tail = (
            f" No detector raised a material concern; {', '.join(clean_bits)} all scored in the clean range."
            if clean_bits
            else " No detector raised a material concern."
        )
        skipped = [DETECTOR_LABELS[n] for n, d in breakdown.items() if d.status == "skipped"]
        if skipped:
            tail += f" Note that {', '.join(skipped)} did not run because the required input was not supplied."
        return lead + tail

    clauses: List[str] = []
    ordered = sorted(
        [(n, d) for n, d in breakdown.items() if d.status == "ok" and d.score >= 0.4],
        key=lambda t: t[1].score * t[1].confidence,
        reverse=True,
    )
    for name, d in ordered[:3]:
        head = d.reasons[0] if d.reasons else f"{DETECTOR_LABELS[name]} flagged this submission"
        head = head[0].lower() + head[1:] if head and head[0].isupper() and not head.startswith(("PAN", "EXIF", "AI", "SDXL", "No")) else head
        clauses.append(f"{head} ({_confidence_phrase(d)})")

    if clauses:
        body = " The decision was driven by " + ("; ".join(clauses)) + "."
    else:
        body = " The decision was driven by the combination of weak signals across detectors rather than any single strong one."

    extra = ""
    pattern_label = PATTERN_LABELS.get(attack_pattern)
    if pattern_label and attack_pattern != "clean":
        extra += f" The pattern matches a known attack class: {pattern_label}."
    if generator:
        extra += f" Spectral fingerprinting attributes the synthetic imagery to {generator}."
    behav = breakdown.get("behavioral")
    if behav is not None and behav.status == "ok" and behav.score >= 0.5:
        seen = (behav.signals.get("evidence") or {}).get("events_seen")
        extra += (
            " Behavioral biometrics scored this fill "
            f"{behav.score:.0%} anomalous"
            + (f" over {seen} captured interaction events" if seen else "")
            + " — that signal is about the person at the keyboard, not the files they uploaded."
        )
    elif behav is not None and behav.status == "skipped":
        extra += (
            " No form-fill telemetry accompanied this packet, so behavioral biometrics did not run;"
            " the verdict rests on the uploaded artifacts alone."
        )
    if linkage:
        links = linkage.get("asset_links", []) or []
        n_face = len(linkage.get("face_links", []) or [])
        n_exact = sum(1 for m in links if m.get("match") == "exact")
        n_near = len(links) - n_exact
        if n_face or links:
            parts = [f"{n_face} face match(es)"]
            if n_exact:
                parts.append(f"{n_exact} identical asset(s)")
            if n_near:
                parts.append(f"{n_near} near-identical asset(s)")
            extra += f" Cross-submission lookup found {', '.join(parts)} in prior submissions."

    caveat = ""
    errored = [DETECTOR_LABELS[n] for n, d in breakdown.items() if d.status == "error"]
    if errored:
        caveat += f" {', '.join(errored)} failed to run, so this verdict rests on partial evidence."
    if abstained:
        caveat += (
            " The fused score fell inside the abstention band around the decision threshold, so Verityne "
            "declined to decide automatically and routed the case to a human."
        )
    return lead + body + extra + caveat
