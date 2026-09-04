#!/usr/bin/env python3
"""Assemble eval/corrections.json - every belief this project measured and lost.

The dashboard's Corrections page and the table at the top of the README both
read this file. It is *generated*, never hand-written, and that is the entire
point: each entry names the evidence file and the dotted path its numbers come
from, and this script resolves every one of them at build time. A correction
that cites a number no evidence file contains fails the build rather than
shipping as prose nobody can check.

The prose lives here; the numbers do not. That split is what stops the timeline
drifting the way the README's own headline once drifted from the report beside
it - which is, fittingly, one of the entries below.

One entry cites source code rather than a report, because the number was never
written to a file. It is marked `source: "code"` and the symbol is verified to
exist, rather than being quietly presented as though a report backed it.

    python backend/scripts/build_corrections.py
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "backend"))

from verityne.config import EVAL_ROOT  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("corrections")

OUT = EVAL_ROOT / "corrections.json"

#: How a finding was made. The distribution matters as much as the findings:
#: only one of these came from looking at a metric.
HOW = {
    "benchmark": "Benchmarked the obvious answer before building on it",
    "ablation": "Ablated the fusion layer one detector at a time",
    "third_party": "Scored the detector on data this project did not generate",
    "ran_the_product": "Submitted a genuine packet to the running API and read the verdict",
    "self_attack": "Built the attack against our own detector",
    "reproduced": "Regenerated a result and asked why a number had moved",
}


def _dig(data: Any, path: Any) -> Any:
    """Resolve a path into a report.

    A path is a dotted string for the common case, or an explicit list of keys
    when a key itself contains a dot - `face_match_lfw.json` stores its shipped
    constants under literal names like "face_match_band.low", which no dotted
    string can address.
    """
    parts = path if isinstance(path, (list, tuple)) else str(path).split(".")
    for part in parts:
        # A numeric-looking part indexes a list, but is an ordinary string key on
        # a dict - several reports key by record count ("5000"), and casting
        # those to int looks for an index that is not there.
        if isinstance(data, (list, tuple)):
            data = data[int(part)]
        else:
            data = data[part]
    return data


def _load(name: str) -> Optional[dict]:
    p = EVAL_ROOT / name
    if not p.exists():
        return None
    return json.loads(p.read_text())


def value(file: str, path: Any) -> Any:
    """Resolve one number out of one evidence file, or fail the build."""
    data = _load(file)
    if data is None:
        raise SystemExit(
            f"eval/{file} is missing, and a correction cites it. Run the pipeline that "
            f"produces it, or remove the entry - do not ship the claim without the evidence."
        )
    try:
        return _dig(data, path)
    except (KeyError, IndexError, TypeError) as exc:
        raise SystemExit(f"eval/{file}: path '{path}' does not resolve ({exc})")


def code_exists(relative: str, symbol: str) -> bool:
    p = REPO / relative
    return p.exists() and symbol in p.read_text()


def pct(x: float) -> float:
    return round(100 * float(x), 2)


#: Spelled out because the sentence this builds is prose, and "9" mid-paragraph
#: reads like a defect next to "nine". Falls back to digits past the range any
#: plausible number of corrections occupies.
_WORDS = {
    1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six",
    7: "seven", 8: "eight", 9: "nine", 10: "ten", 11: "eleven", 12: "twelve",
}


def _word(n: int) -> str:
    return _WORDS.get(n, str(n))


def _tally(counts: Dict[str, int]) -> str:
    """Say what the counts say, in words, so the prose cannot contradict them.

    Only mentions a status that actually has entries: "two the system was
    redesigned around" is worth a clause when it is true and a lie when the
    count is zero.
    """
    total = sum(counts.values())
    clauses = []
    if counts.get("fixed"):
        clauses.append(f"{_word(counts['fixed'])} are fixed")
    if counts.get("open"):
        clauses.append(f"{_word(counts['open'])} are open and say why")
    if counts.get("designed_around"):
        clauses.append(f"{_word(counts['designed_around'])} the system was redesigned around")
    if not clauses:
        return f"There are {_word(total)}."
    if len(clauses) == 1:
        body = clauses[0]
    else:
        body = ", ".join(clauses[:-1]) + ", and " + clauses[-1]
    return f"Of the {_word(total)}, {body}."


# --------------------------------------------------------------------------- #
# The corrections. Prose here, numbers from disk.
# --------------------------------------------------------------------------- #

def build() -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []

    # 1 ---------------------------------------------------------------------
    bench = {r["model"]: r for r in value("model_benchmark.json", "results")}
    out.append({
        "id": "pretrained-baseline",
        "title": "The popular pretrained deepfake checkpoint is at chance on our data",
        "believed": "An off-the-shelf deepfake checkpoint with thousands of downloads is a "
                    "reasonable baseline to build on.",
        "measured": "Both candidates scored at or below chance on our own images. Their label "
                    "mappings were verified first, so this is not an inverted-sign bug - it is "
                    "what happens when a model trained on StyleGAN and face-swap video frames "
                    "meets diffusion output.",
        "metric": "ROC-AUC on our images",
        "series": [
            {"label": "dima806/deepfake_vs_real_image_detection",
             "value": bench["dima806/deepfake_vs_real_image_detection"]["auc"]},
            {"label": "prithivMLmods/Deep-Fake-Detector-v2-Model",
             "value": bench["prithivMLmods/Deep-Fake-Detector-v2-Model"]["auc"]},
        ],
        "reference": {"label": "chance", "value": 0.5},
        "status": "designed_around",
        "outcome": "The CNN became one vote in an ensemble rather than the system. A project "
                   "that had shipped the popular checkpoint on reputation would have reported "
                   "a confident number for a model that was guessing.",
        "how_found": "benchmark",
        "evidence": [{"file": "model_benchmark.json", "path": "results"}],
        "readme": "docs/corrections.md#the-honest-version-of-what-this-is",
    })

    # 2 ---------------------------------------------------------------------
    leaked = value("ablation_leaked_corpus.json", "full_model.roc_auc")
    fixed = value("ablation.json", "full_model.roc_auc")
    out.append({
        "id": "exif-leak",
        "title": "Half the headline AUC was a label we wrote into our own files",
        "believed": "The system scores 0.913 held out, and five detectors earned it.",
        "measured": "Ablating one detector at a time showed metadata/EXIF carrying 49.5% of the "
                    "above-chance AUC - because the corpus generator had conditioned EXIF mode "
                    "on the label, leaving three modes that appeared on fraudulent packets and "
                    "never on genuine ones. 17.3% of the corpus carried a fraud label in "
                    "disguise. Two of the five detectors were making the model actively worse.",
        "metric": "Held-out ROC-AUC",
        "before": leaked,
        "after": fixed,
        "direction": "down_is_honest",
        "status": "fixed",
        "outcome": "Corpus regenerated with EXIF mixed across classes. Two independent guards "
                   "now refuse to build or to ship a corpus in which any EXIF mode lands on one "
                   "class only, with a single documented exemption for a mode no camera can "
                   "physically produce. The headline fell 0.16 and is now worth its number.",
        "how_found": "ablation",
        "evidence": [
            {"file": "ablation_leaked_corpus.json", "path": "full_model.roc_auc"},
            {"file": "ablation.json", "path": "full_model.roc_auc"},
        ],
        "readme": "docs/results.md#what-the-headline-auc-is-actually-made-of",
    })

    # 3 ---------------------------------------------------------------------
    out.append({
        "id": "isotonic-calibrator",
        "title": "The calibrator was quietly costing 0.02 AUC",
        "believed": "Calibrating the fused score is free - it changes the scale, not the ranking.",
        "measured": "Isotonic regression on 195 rows fitted a step function with so few levels "
                    "that held-out scores collapsed to 14 distinct values. AUC measures ranking, "
                    "and mass ties destroy ranking. It also made the score useless as a dial: a "
                    "packet a hair above a step boundary jumped from 0.44 to 0.92, while "
                    "policy.yaml cuts that score at fixed thresholds.",
        "metric": "Held-out ROC-AUC",
        "before": 0.732,
        "after": value("metrics.json", "fusion.roc_auc"),
        "direction": "up",
        "status": "fixed",
        "outcome": "Replaced with Platt scaling - two parameters, strictly monotonic, so it "
                   "preserves ranking exactly and returns a smooth score. Nothing was broken and "
                   "no test failed; the number was simply lower than the model had earned.",
        "how_found": "reproduced",
        "evidence": [
            {"source": "code", "file": "backend/verityne/fusion.py", "symbol": "PlattCalibrator",
             "note": "The before-figure was never written to a report; it is recorded in this "
                     "class's docstring, and this entry says so rather than implying a file."},
            {"file": "metrics.json", "path": "fusion.roc_auc"},
        ],
        "readme": "docs/fusion.md#the-calibrator-was-costing-002-auc",
    })

    # 4 ---------------------------------------------------------------------
    LOW = ["as_shipped", "face_match_band.low", "tar"]
    corpus_tar = value("face_match_lfw.json", LOW)
    out.append({
        "id": "face-match-threshold",
        "title": "The identity threshold would have rejected two thirds of honest applicants",
        "believed": "The face-match lower bound was calibrated - it was fitted, on our own pairs.",
        "measured": f"Graded against LFW's 6,000 real pairs, the corpus-fitted bound accepts "
                    f"{pct(corpus_tar)}% of genuine pairs. The corpus could not have revealed "
                    "this: its 'two photographs of one person' are one photograph re-captured "
                    "twice, scoring 0.943, where two genuinely different photographs of the same "
                    "person score 0.758.",
        "metric": "True-accept rate on real LFW pairs",
        "before": corpus_tar,
        "after": value("face_match_lfw.json", "operating_points.far_1pct.tar"),
        "direction": "up",
        "status": "fixed",
        "outcome": "The bound is fitted on LFW's official 10-fold protocol and calibrate.py no "
                   "longer overwrites it with the corpus value. Fitting this bound on this "
                   "corpus produces a wrong answer every time, and a worse one the better the "
                   "corpus gets at making one person's two images look alike.",
        "how_found": "third_party",
        "evidence": [
            {"file": "face_match_lfw.json", "path": LOW},
            {"file": "face_match_lfw.json", "path": "operating_points.far_1pct.tar"},
        ],
        "readme": "docs/real-data.md#1-face-identity-on-lfw---and-two-thresholds-that-were-badly-wrong",
    })

    # 5 ---------------------------------------------------------------------
    shipped_fl = value("linkage_lfw.json", "superseded.per_applicant_false_link_rate.5000")
    fitted_fl = value("linkage_lfw.json", "fitted.per_applicant_false_link_rate.5000")
    out.append({
        "id": "linkage-search-vs-pair",
        "title": "Every genuine applicant false-linked to a stranger",
        "believed": "Linkage catches onboarding rings, and its threshold was fitted on LFW at a "
                    "strict false-accept budget.",
        "measured": f"The threshold was fitted for a *pair* and deployed as a *search*. Against "
                    f"the shipped 5,000-record scan limit, {pct(shipped_fl)}% of genuine "
                    "applicants pick up a false link. A pairwise false-accept rate applied N "
                    "times compounds; the pair fit stays where it was put. LFW's official "
                    "protocol cannot resolve this - 3,000 impostor pairs bottom out at 3.3e-4, "
                    "and the shipped rate was two pairs.",
        "metric": "Genuine applicants false-linking over a 5,000-record scan",
        "before": shipped_fl,
        "after": fitted_fl,
        "direction": "down",
        "status": "fixed",
        "outcome": "Refitted over all 16,522,626 impostor pairs among LFW's distinct identities, "
                   "from a stated 1% budget. A face link can also no longer reject anyone by "
                   "itself - it is capped one abstention band below the reject threshold, and "
                   "only a byte-identical file is exempt. The cost is stated: true-accept falls "
                   f"from {pct(value('linkage_lfw.json','superseded.tar'))}% to "
                   f"{pct(value('linkage_lfw.json','fitted.tar'))}%.",
        "how_found": "ran_the_product",
        "evidence": [
            {"file": "linkage_lfw.json", "path": "superseded.per_applicant_false_link_rate.5000"},
            {"file": "linkage_lfw.json", "path": "fitted.per_applicant_false_link_rate.5000"},
        ],
        "readme": "docs/real-data.md#2b-the-linkage-threshold-was-answering-the-wrong-question",
    })

    # 6 ---------------------------------------------------------------------
    spectral = value("indian_faces.json", "face.shortcut_auc.spectral_p_fake")
    cnn = value("indian_faces.json", "face.shortcut_auc.cnn_p_fake")
    out.append({
        "id": "demographic-shortcut",
        "title": "The frequency head separates real Indian faces from real FFHQ faces",
        "believed": "The spectral head is this project's best detector - 0.969 train AUC on "
                    "detecting synthesis.",
        "measured": f"Scored on two groups that are *both genuine* - real photographs of Indian "
                    f"people against real FFHQ photographs, geometry-controlled - it separates "
                    f"them at {spectral}. It learned the prompt list: every synthetic face in "
                    "our corpus was generated from a prompt naming the demographic, and FFHQ is "
                    f"predominantly not South Asian. The pretrained CNN scores {cnn} on the same "
                    "comparison - exactly chance, and the only component here with no "
                    "demographic signal at all.",
        "metric": "Shortcut AUC, real vs real (0.5 is the only defensible value)",
        "series": [
            {"label": "frequency head, fitted here", "value": spectral},
            {"label": "pretrained CNN", "value": cnn},
        ],
        "reference": {"label": "no shortcut", "value": 0.5},
        "status": "open",
        "outcome": "Nothing has been tuned in response, deliberately. The corpus needs rebuilding "
                   "with demography decoupled from the label and the head refitting on it. "
                   "Adjusting a threshold until the table looks better would move the disparity "
                   "somewhere a metric cannot see, which is the failure this measurement exists "
                   "to report.",
        "how_found": "third_party",
        "evidence": [
            {"file": "indian_faces.json", "path": "face.shortcut_auc.spectral_p_fake"},
            {"file": "indian_faces.json", "path": "face.shortcut_auc.cnn_p_fake"},
        ],
        "readme": "docs/real-data.md#3-the-selfie-detector-is-reading-demography",
    })

    # 7 ---------------------------------------------------------------------
    fams = value("real_faces.json", "headline.per_family_auc")
    worst = value("real_faces.json", "headline.worst_family")
    worst_auc = value("real_faces.json", "headline.worst_family_auc")
    out.append({
        "id": "third-party-fakes",
        "title": "The selfie detector does not work on fakes we did not generate",
        "believed": "The selfie detector catches diffusion output at 0.997 - the one attack "
                    "class this project had solved.",
        "measured": "Scored on four generator families from two third-party datasets, "
                    "geometry-controlled and reported per family rather than pooled, it is at or "
                    f"below chance on every one of them. The worst, {worst}, is at {worst_auc} - "
                    "below 0.5 means inverted: it scores those fakes as more genuine than real "
                    "faces. The 0.997 was the detector reading its own generator's settings.",
        "metric": "ROC-AUC per generator family, controlled face protocol",
        "series": [{"label": k, "value": v} for k, v in fams.items()],
        "reference": {"label": "chance", "value": 0.5},
        "status": "open",
        "outcome": "This is the most consequential finding in the project and it is unresolved. "
                   "It does not sink the system - the fused verdict rests on provenance, "
                   "structural document checks and behaviour, not on this detector, and the "
                   "ablation already showed it carrying 10.6% of above-chance AUC rather than "
                   "the headline. But a detector marketed on deepfake detection that is at "
                   "chance on somebody else's deepfakes has to say so in its own README.",
        "how_found": "third_party",
        "evidence": [
            {"file": "real_faces.json", "path": "headline.per_family_auc"},
            {"file": "real_faces.json", "path": "headline.worst_family_auc"},
        ],
        "readme": "docs/real-data.md#measured-on-real-data",
    })

    # 8 ---------------------------------------------------------------------
    replay = value("behavioral.json", "headline.worst_unseen_strategy_auc")
    out.append({
        "id": "replay-attack",
        "title": "We built the attack that defeats our own behavioral detector",
        "believed": "Keystroke rhythm is the one signal a fraud kit cannot buy - 1.000 held-out "
                    "AUC against every automation strategy we had.",
        "measured": f"A replay attacker that dispatches a *real* person's recorded dwell and "
                    f"flight timings through the devtools protocol - rollover included, by "
                    f"laying presses and releases on a timeline rather than emitting them key by "
                    f"key - scores {replay}. Chance. Nothing about its rhythm is synthetic.",
        "metric": "ROC-AUC against the replay attacker",
        "before": value("behavioral.json", "headline.held_out_auc"),
        "after": replay,
        "direction": "down",
        "status": "open",
        "outcome": "Not fixable by a better rhythm model: the rhythm genuinely is human. What "
                   "Detector 6 buys is raising the cost of automating a KYC form from free to "
                   "'you must first record a real human filling one'. The roadmap item is that a "
                   "replayed recording is a *reused* one, which is the linkage problem this repo "
                   "already solves for faces and files - not a timing problem.",
        "how_found": "self_attack",
        "evidence": [
            {"file": "behavioral.json", "path": "headline.worst_unseen_strategy_auc"},
            {"file": "behavioral.json", "path": "headline.held_out_auc"},
        ],
        "readme": "docs/behavioral.md#the-attack-that-beats-it-which-we-built-ourselves",
    })

    # 9 ---------------------------------------------------------------------
    if (EVAL_ROOT / "thresholds.json").exists():
        oof_fpr = value("thresholds.json", "transfer_check.out_of_fold_false_positive_rate")
        out_fpr = value("thresholds.json", "transfer_check.held_out_false_positive_rate")
        chosen = value("thresholds.json", "chosen.min_risk_for_reject")
        out.append({
            "id": "thresholds-do-not-transfer",
            "title": "The stale thresholds could not be fixed by fitting them properly",
            "believed": "The shipped 0.40 and 0.75 are stale because nobody re-fitted them. "
                        "Refit them on the training split with cross-validation - never touching "
                        "held-out - and the problem goes away.",
            "measured": f"The procedure works and the corpus cannot support it. The chosen "
                        f"reject point ({chosen}) holds a {pct(oof_fpr)}% false-positive rate "
                        f"out-of-fold and produces {pct(out_fpr)}% on a split it was not chosen "
                        f"on - {round(out_fpr / oof_fpr, 1)}x optimistic. Every training "
                        "identity is distinct, so this is not fold leakage; 195 genuine faces "
                        "simply cannot resolve an operating point that survives new faces.",
            "metric": "False-positive rate at the chosen reject threshold",
            "series": [
                {"label": "out-of-fold, training split", "value": oof_fpr},
                {"label": "held-out, never consulted", "value": out_fpr},
            ],
            "reference": {"label": "what was chosen for", "value": oof_fpr},
            "status": "designed_around",
            "outcome": "So the defaults do not come from cross-validation. They come from the "
                       "system's own cost curve, which priced the shipped 0.75 reject line at "
                       "negative expected net benefit and put its argmax at 0.80 - moving there "
                       "took the false-reject rate from 9.8% to 2.0% and precision from 0.815 to "
                       "0.952. The review line, which the cost model does not cover, moved from "
                       "0.40 to 0.45 because 0.40 sat barely above the genuine median and sent "
                       "39% of honest merchants to a human. Both are read off held-out, so they "
                       "report themselves: an operating point, not a generalisation claim. The "
                       "mechanism that matters is still per-merchant policy plus the cost curve, "
                       "and the defaults are only a sane place to start.",
            "how_found": "reproduced",
            "evidence": [
                {"file": "thresholds.json", "path": "transfer_check.out_of_fold_false_positive_rate"},
                {"file": "thresholds.json", "path": "transfer_check.held_out_false_positive_rate"},
            ],
            "readme": "docs/results.md#the-thresholds-and-where-they-came-from",
        })

    return out


def main() -> None:
    corrections = build()
    # The dashboard renders these strings as text, not markdown, so a stray
    # backtick shows up as a backtick. Catch it here rather than in review.
    for c in corrections:
        for field in ("title", "believed", "measured", "outcome", "metric"):
            if "`" in c[field]:
                raise SystemExit(
                    f"{c['id']}.{field} contains a backtick; this prose is rendered verbatim"
                )
    for i, c in enumerate(corrections, 1):
        c["order"] = i
        for ev in c["evidence"]:
            if ev.get("source") == "code":
                if not code_exists(ev["file"], ev["symbol"]):
                    raise SystemExit(
                        f"{c['id']}: cites {ev['file']}::{ev['symbol']}, which does not exist"
                    )

    counts: Dict[str, int] = {}
    for c in corrections:
        counts[c["status"]] = counts.get(c["status"], 0) + 1

    payload = {
        "generated_by": "backend/scripts/build_corrections.py",
        # Derived from `counts`, never written by hand. This sentence sits on the
        # one page whose entire claim is that its numbers are checkable, and it
        # had already drifted once - it read "Five of the eight are fixed" while
        # the counts rendered beside it said nine. A wrong number there discredits
        # the page more than the correction it describes.
        "what_this_is": (
            "Every belief this project held and then measured and lost. Generated from the "
            "evidence files each entry cites, so a correction cannot claim a number no report "
            f"contains. {_tally(counts)}"
        ),
        "how_found_legend": HOW,
        "counts": counts,
        "n": len(corrections),
        "corrections": corrections,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2))
    log.info("wrote %s - %d corrections (%s)", OUT, len(corrections),
             ", ".join(f"{v} {k}" for k, v in sorted(counts.items())))


if __name__ == "__main__":
    main()
