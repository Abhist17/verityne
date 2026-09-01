"""Detector 6 - Behavioral Biometrics.

Detectors 1-5 all inspect an *artifact* the applicant uploaded. Every one of
those artifacts is purchasable: a face that does not exist costs four seconds, a
matched PAN + selfie + liveness kit costs a few hundred rupees, and an EXIF block
can be rewritten in a line of Python. What none of those kits ship is the motor
timing of the person at the keyboard. That is what this detector reads.

Three families of signal, none of which touch a pixel:

  * keystroke dynamics - dwell (key-down to key-up) and flight (key-up to the
    next key-down). Both are involuntary, both vary person to person, and both
    have a *variance* that automation does not reproduce: a scripted fill either
    emits no key events at all or emits them on a timer, which shows up as a
    dwell coefficient of variation near zero and flight intervals below the
    ~15 ms floor a finger can physically achieve.
  * pointer and form interaction - real fills curve, overshoot, hesitate on the
    date of birth, and scroll back to fix a typo. Kits fill top to bottom, never
    revisit a field, never press backspace, and paste the ID number in from a
    text file.
  * environment coherence - a browser claiming en-IN from a UTC+3 clock with
    `navigator.webdriver` set is not a merchant in Mumbai.

**On the thresholds.** Unlike detectors 1-5, the constants here are *priors*, not
values fitted on this repository's data - we have no labelled corpus of genuine
Indian merchant form-fills, and inventing one in a synthesiser would be fitting
to our own assumptions. They are set conservatively, in the direction that costs
a false accept rather than a false reject, and every one of them is a named
constant below so it can be re-fitted the moment real telemetry exists. The
detector reports low confidence when it has seen too little to judge, and
`applicable()` returns False when the browser sent nothing at all, so a
submission with no telemetry is *skipped*, never silently scored as clean.
"""
from __future__ import annotations

import functools
import logging
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

log = logging.getLogger("verityne.behavioral")

from ..config import MODEL_ROOT
from ..schemas import DetectorOutput
from ..utils.keystroke import feature_row, keystroke_features
from .base import Detector, SubmissionPayload, clamp

# --------------------------------------------------------------------------- #
# Priors. Every threshold a rule fires on lives here, named, so re-fitting on
# real telemetry is a diff to this block and nothing else.
# --------------------------------------------------------------------------- #

#: Shortest gap between releasing one key and pressing the next that a human
#: hand achieves. Rollover on adjacent fingers gets into the low tens of ms;
#: below this the events were generated, not typed.
HUMAN_FLIGHT_FLOOR_MS = 15.0
#: Dwell time is involuntary and noisy. A coefficient of variation this low means
#: every key was held for the same number of milliseconds - a timer, not a hand.
ROBOTIC_DWELL_CV = 0.08
#: Same argument for the gaps between keys.
ROBOTIC_FLIGHT_CV = 0.10
#: Sustained characters per second above this is not typing. Fast human touch
#: typists peak around 8-10 cps in burst and far less on a form with tab stops.
IMPLAUSIBLE_TYPING_CPS = 14.0
#: A full KYC form - name, ID number, date of birth, address - read and typed by
#: a human who is being careful about their own money. Below this nobody read it.
MIN_PLAUSIBLE_FORM_SECONDS = 15.0
#: Kits that pad with `sleep(N)` land on a round number to within a rounding
#: error. Humans never do.
SLEEP_ROUND_NUMBERS_S = (30.0, 60.0, 90.0, 120.0, 180.0, 240.0, 300.0, 600.0)
SLEEP_TOLERANCE_S = 0.75
#: Straightness = |displacement| / path length. A hand-driven pointer wanders;
#: 1.0 is a straight line, which is what a synthetic move emits.
STRAIGHT_LINE_RATIO = 0.97
#: Below this many samples the pointer statistics are not worth reading.
MIN_MOUSE_SAMPLES = 8
#: Real mousemove delivery is irregular - compositor scheduling, hand speed.
#: A near-constant sampling interval is a generated path.
ROBOTIC_MOUSE_DT_CV = 0.12
#: Normalised direction-change entropy below this is a path with no tremor.
LOW_PATH_ENTROPY = 0.35
#: Enough keystrokes to read a rhythm from at all.
MIN_KEYS_FOR_RHYTHM = 12
#: Below this the fitted model is not saying anything a rule is not already
#: saying more legibly, and a low-probability hit would only add noise to the
#: noisy-OR. Set at the model's own operating point.
MODEL_ALERT_P = 0.5
#: Enough keystrokes to read one confidently.
CONFIDENT_KEYS = 35

#: Fields where a paste is worth a rule on its own. A merchant knows their own
#: PAN; they do not keep it in a text file to copy from. (Address and email are
#: deliberately absent - pasting those is ordinary behaviour.)
SENSITIVE_FIELDS = {"pan", "id_number", "claimed_id_number", "aadhaar", "dob", "claimed_dob", "account_number"}

#: Coarse region -> plausible UTC offset in minutes, used only to catch a
#: contradiction between the language a browser advertises and the clock it
#: keeps. Deliberately coarse and deliberately incomplete: a rule only fires on
#: a region listed here, so an unlisted locale is never penalised for being
#: unusual. Diaspora is real - an en-IN browser in London is an ordinary human -
#: so this needs a *large* separation before it says anything, and even then it
#: contributes a soft weight, never a rejection.
REGION_OFFSET_MIN = {
    "IN": 330, "PK": 300, "BD": 360, "LK": 330, "NP": 345,
    "GB": 0, "IE": 0, "PT": 0,
    "DE": 60, "FR": 60, "ES": 60, "IT": 60, "PL": 60, "NL": 60, "SE": 60,
    "RU": 180, "TR": 180, "SA": 180, "AE": 240, "IR": 210,
    "CN": 480, "SG": 480, "MY": 480, "PH": 480, "HK": 480,
    "JP": 540, "KR": 540, "AU": 600, "NZ": 720,
    "US": -300, "CA": -300, "BR": -180, "MX": -360,
    "NG": 60, "ZA": 120, "KE": 180, "EG": 120,
}
#: How far a browser's clock may sit from its language's home region before the
#: pair is called incoherent. Four hours clears every daylight-saving shift and
#: every "I am travelling" case inside a continent.
LOCALE_OFFSET_TOLERANCE_MIN = 4 * 60


#: Fitted by `scripts/train_behavioral.py` on real Aalto typing and real
#: browser-automation sessions. Absent until that script has been run, and the
#: rules below stand alone when it is - the detector is never dead on arrival.
KEYSTROKE_MODEL_PATH = MODEL_ROOT / "behavioral_keystroke.joblib"


@functools.lru_cache(maxsize=1)
def _keystroke_model():
    if not KEYSTROKE_MODEL_PATH.exists():
        return None
    try:
        import joblib

        return joblib.load(KEYSTROKE_MODEL_PATH)
    except Exception as exc:  # noqa: BLE001
        log.warning("keystroke model unreadable: %s", exc)
        return None


def reload_keystroke_model() -> None:
    _keystroke_model.cache_clear()


def model_p_automated(feats: Dict[str, Any]) -> Optional[float]:
    """P(this rhythm was generated), from the fitted model, or None.

    None rather than 0.5 when the model is missing or the sample is too thin:
    the caller must be able to tell "the model says human" from "there was no
    model", because those two justify completely different weights.
    """
    bundle = _keystroke_model()
    if bundle is None:
        return None
    if int(feats.get("keystroke_count") or 0) < int(bundle.get("min_keys", MIN_KEYS_FOR_RHYTHM)):
        return None
    try:
        row = feature_row(feats, bundle["feature_names"])
        return float(bundle["model"].predict_proba(row[None, :])[0, 1])
    except Exception as exc:  # noqa: BLE001
        log.warning("keystroke model failed, falling back to rules: %s", exc)
        return None


def _stats(xs: Sequence[float]) -> Tuple[float, float, float]:
    """(mean, std, coefficient of variation). CV is 0.0 for an empty sample."""
    a = np.asarray([x for x in xs if x is not None and np.isfinite(x)], dtype=float)
    if a.size == 0:
        return 0.0, 0.0, 0.0
    mean = float(a.mean())
    std = float(a.std())
    return mean, std, float(std / mean) if mean > 1e-9 else 0.0


def _screen_dims(raw: Any) -> Optional[Tuple[int, int]]:
    """Screen size out of whatever shape the page sent it in.

    Accepts `[w, h]` and `{"w":…, "h":…}` / `{"width":…, "height":…}`. Being
    liberal here is the point: this is the untrusted boundary, the shipped
    collector sends the object form, and a shape this function did not recognise
    would not raise - it would silently retire the headless-browser check, which
    is the worst of the three outcomes.
    """
    if isinstance(raw, dict):
        w = raw.get("w", raw.get("width"))
        h = raw.get("h", raw.get("height"))
    elif isinstance(raw, (list, tuple)) and len(raw) == 2:
        w, h = raw
    else:
        return None
    try:
        return int(w), int(h)
    except (TypeError, ValueError):
        return None


def _region_of(tag: str) -> Optional[str]:
    """'en-IN' -> 'IN'. Returns None for a bare language tag like 'en'."""
    parts = str(tag).replace("_", "-").split("-")
    for p in parts[1:]:
        if len(p) == 2 and p.isalpha():
            return p.upper()
    return None


# --------------------------------------------------------------------------- #
# Feature extraction
# --------------------------------------------------------------------------- #

def extract_features(events: Dict[str, Any]) -> Dict[str, Any]:
    """Turn one raw telemetry buffer into the flat feature row the scorer reads.

    Pure and total: a malformed or half-empty buffer yields zeros and empty
    counts rather than raising, because this runs on attacker-controlled input.
    All timestamps arriving from the browser are milliseconds from form load
    (`performance.now()`), which is monotonic and immune to a clock change
    mid-fill.
    """
    f: Dict[str, Any] = {}

    keys = [k for k in (events.get("keys") or []) if isinstance(k, dict)]
    mouse = [m for m in (events.get("mouse") or []) if isinstance(m, dict)]
    focus = [x for x in (events.get("focus") or []) if isinstance(x, dict)]
    pastes = [p for p in (events.get("paste") or []) if isinstance(p, dict)]
    env = events.get("env") or {}

    # ---- keystroke dynamics -------------------------------------------------
    dwells, downs, printable, backspaces = [], [], 0, 0
    for k in keys:
        down, up = k.get("down"), k.get("up")
        if isinstance(down, (int, float)):
            downs.append(float(down))
            if isinstance(up, (int, float)) and up >= down:
                dwells.append(float(up) - float(down))
        name = str(k.get("key") or "")
        if name == "Backspace":
            backspaces += 1
        elif len(name) == 1:
            printable += 1

    downs.sort()
    # Flight is key-up to the next key-down. We only have ordered downs and per
    # key dwell, so reconstruct it as (next down - this down) - this dwell, which
    # is the same quantity and survives events arriving out of order.
    flights: List[float] = []
    ordered = sorted(
        (k for k in keys if isinstance(k.get("down"), (int, float))),
        key=lambda k: float(k["down"]),
    )
    for a, b in zip(ordered, ordered[1:]):
        up_a = a.get("up")
        end_a = float(up_a) if isinstance(up_a, (int, float)) and up_a >= a["down"] else float(a["down"])
        flights.append(float(b["down"]) - end_a)
    flights = [x for x in flights if -50.0 < x < 5000.0]  # drop pauses and clock glitches

    d_mean, _, d_cv = _stats(dwells)
    fl_mean, _, fl_cv = _stats([x for x in flights if x >= 0])


    positive_flights = [x for x in flights if x >= 0]

    # The fitted model's inputs, computed by the same function the training
    # script uses. Merged in rather than recomputed here so the row a model was
    # fitted on and the row it is served cannot drift apart - see
    # utils/keystroke.py, which exists entirely for that reason.
    span_s = (max(downs) - min(downs)) / 1000.0 if len(downs) >= 2 else None
    f.update(keystroke_features(
        dwells, flights, n_keys=len(keys), backspaces=backspaces,
        span_s=span_s, printable=printable,
    ))

    f["keystroke_count"] = len(keys)
    f["dwell_samples"] = len(dwells)
    f["dwell_mean_ms"] = round(d_mean, 2)
    f["dwell_cv"] = round(d_cv, 4)
    f["flight_samples"] = len(positive_flights)
    f["flight_mean_ms"] = round(fl_mean, 2)
    f["flight_cv"] = round(fl_cv, 4)
    f["flight_min_ms"] = round(float(min(flights)), 2) if flights else 0.0
    f["backspace_rate"] = round(backspaces / len(keys), 4) if keys else 0.0

    span_s = (max(downs) - min(downs)) / 1000.0 if len(downs) >= 2 else 0.0
    f["typing_speed_cps"] = round(printable / span_s, 3) if span_s > 0.5 else 0.0

    # ---- pointer ------------------------------------------------------------
    pts = [
        (float(m["t"]), float(m["x"]), float(m["y"]))
        for m in mouse
        if all(isinstance(m.get(c), (int, float)) for c in ("t", "x", "y"))
    ]
    pts.sort(key=lambda p: p[0])
    f["mouse_points"] = len(pts)
    f["mouse_straightness"] = 0.0
    f["mouse_path_entropy"] = 0.0
    f["mouse_moving_segments"] = 0
    f["mouse_dt_samples"] = 0
    f["mouse_dt_cv"] = 0.0
    f["mouse_jerk_cv"] = 0.0

    if len(pts) >= MIN_MOUSE_SAMPLES:
        arr = np.asarray(pts, dtype=float)
        t, xy = arr[:, 0], arr[:, 1:]
        seg = np.diff(xy, axis=0)
        seg_len = np.hypot(seg[:, 0], seg[:, 1])
        path = float(seg_len.sum())
        disp = float(np.hypot(*(xy[-1] - xy[0])))
        f["mouse_straightness"] = round(disp / path, 4) if path > 1e-6 else 1.0

        moving = seg_len > 1e-6
        f["mouse_moving_segments"] = int(moving.sum())
        if moving.sum() >= 4:
            ang = np.arctan2(seg[moving, 1], seg[moving, 0])
            hist, _ = np.histogram(ang, bins=16, range=(-math.pi, math.pi))
            p = hist[hist > 0] / hist.sum()
            # `+ 0.0` normalises the -0.0 that a single-bin path produces, so a
            # perfectly straight line compares as the zero it actually is.
            f["mouse_path_entropy"] = round(float(-(p * np.log(p)).sum() / math.log(16)) + 0.0, 4)

        dt = np.diff(t)
        dt = dt[dt > 0]
        f["mouse_dt_samples"] = int(dt.size)
        if dt.size >= 4:
            f["mouse_dt_cv"] = round(_stats(dt)[2], 4)
            speed = seg_len[: dt.size] / dt
            if speed.size >= 3:
                f["mouse_jerk_cv"] = round(_stats(np.abs(np.diff(speed, n=2)))[2], 4)

    # ---- form interaction ---------------------------------------------------
    f["paste_count"] = len(pastes)
    f["paste_into_sensitive"] = sum(1 for p in pastes if str(p.get("field", "")).lower() in SENSITIVE_FIELDS)

    seen: List[str] = []
    revisits = 0
    for ev in focus:
        name = str(ev.get("field") or "")
        if not name:
            continue
        if name in seen:
            revisits += 1
        else:
            seen.append(name)
    f["fields_touched"] = len(seen)
    f["field_revisits"] = revisits

    # Was the form filled strictly in DOM order, never going back? The browser
    # sends each field's declared index; a human's order is almost never a clean
    # ascending run with zero revisits.
    idx = [ev.get("index") for ev in focus if isinstance(ev.get("index"), int)]
    f["field_order_monotonic"] = float(bool(idx) and idx == sorted(idx) and revisits == 0)

    hold = [
        float(ev["out"]) - float(ev["in"])
        for ev in focus
        if isinstance(ev.get("in"), (int, float)) and isinstance(ev.get("out"), (int, float))
        and float(ev["out"]) >= float(ev["in"])
    ]
    f["field_dwell_cv"] = round(_stats(hold)[2], 4)

    # ---- timing -------------------------------------------------------------
    loaded, submitted = events.get("form_loaded_at"), events.get("submitted_at")
    total = 0.0
    if isinstance(loaded, (int, float)) and isinstance(submitted, (int, float)) and submitted > loaded:
        total = (float(submitted) - float(loaded)) / 1000.0
    elif downs:
        total = (max(downs) - min(downs)) / 1000.0
    f["total_time_s"] = round(total, 2)
    f["near_round_sleep_s"] = next(
        (r for r in SLEEP_ROUND_NUMBERS_S if abs(total - r) <= SLEEP_TOLERANCE_S), 0.0
    )

    # ---- environment --------------------------------------------------------
    langs = [str(x) for x in (env.get("languages") or []) if x]
    # JS getTimezoneOffset() is minutes *behind* UTC, i.e. IST is -330. Flip it
    # so the sign matches REGION_OFFSET_MIN and anyone reading a log.
    raw_off = env.get("timezone_offset_min")
    tz_off = -float(raw_off) if isinstance(raw_off, (int, float)) else None
    f["timezone"] = str(env.get("timezone") or "")
    f["timezone_offset_min"] = tz_off
    f["languages"] = langs[:5]

    regions = [r for r in (_region_of(x) for x in langs) if r]
    declared = str(events.get("declared_country") or env.get("country") or "").upper()[:2]
    if declared:
        regions.append(declared)

    incoherent, locale_detail = 0.0, ""
    known = [(r, REGION_OFFSET_MIN[r]) for r in dict.fromkeys(regions) if r in REGION_OFFSET_MIN]
    if tz_off is not None and known:
        # Coherent if *any* advertised region is compatible with the clock - a
        # multilingual browser should not be penalised for its second language.
        gaps = [abs(tz_off - off) for _, off in known]
        if min(gaps) > LOCALE_OFFSET_TOLERANCE_MIN:
            incoherent = 1.0
            r, off = known[int(np.argmin(gaps))]
            locale_detail = (
                f"browser advertises {r} (UTC{off/60:+.1f}) but its clock reads UTC{tz_off/60:+.1f}"
            )
    f["locale_incoherent"] = incoherent
    f["locale_detail"] = locale_detail

    flags = []
    if env.get("webdriver") is True:
        flags.append("navigator.webdriver")
    dims = _screen_dims(env.get("screen"))
    if dims is not None and (dims[0] <= 0 or dims[1] <= 0):
        flags.append("zero-size screen")
    ua = str(env.get("user_agent") or "").lower()
    if any(tok in ua for tok in ("headlesschrome", "phantomjs", "electron/", "puppeteer", "playwright")):
        flags.append("automation user-agent")
    if env.get("hardware_concurrency") in (0, 1) and "mobi" not in ua:
        flags.append("single-core desktop browser")
    f["automation_flags"] = flags

    f["event_count"] = len(keys) + len(mouse) + len(focus) + len(pastes)
    return f


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #

def score_features(
    f: Dict[str, Any], *, use_model: bool = True
) -> Tuple[float, float, List[str], List[Dict[str, Any]]]:
    """Rule hits -> (score, confidence, reasons, hits).

    `use_model=False` scores on the deterministic rules alone. That is not a
    convenience flag: the rules and the fitted model are separately meaningful
    and separately testable, and a test that means to pin rule behaviour should
    not silently start measuring a model checkpoint that may or may not be on
    disk. `scripts/train_behavioral.py` uses it to report what the model adds
    over the rules rather than assuming it adds anything.

    Combined by noisy-OR, the same way `metadata_exif` combines its rules: each
    independent hit erodes the probability that the fill was human, so several
    weak signals compound without any one of them dominating. A hit is 'hard'
    only when it is physically impossible rather than merely unusual, and without
    a hard hit the score is capped below the reject line - behaviour is a strong
    prior, but a nervous first-time user typing slowly on a borrowed laptop must
    never be rejected by rhythm alone.
    """
    hits: List[Tuple[str, float, str]] = []
    keys = int(f.get("keystroke_count") or 0)

    # ---- fitted model -------------------------------------------------------
    # Consulted alongside the rules rather than instead of them. The rules encode
    # things that are true by physics (a 9 ms flight) or by self-declaration (a
    # webdriver flag); the model encodes the shape of a real typing distribution,
    # which is what the rules could only approximate with hand-set constants. It
    # contributes as one more weighted hit, capped below the categorical tier,
    # because a fitted probability is still a statistical claim.
    p_auto = model_p_automated(f) if use_model else None
    if p_auto is not None and p_auto >= MODEL_ALERT_P:
        hits.append((
            "keystroke:model", min(0.80, float(p_auto)),
            f"A model fitted on 168,595 real people's typing and on real browser "
            f"automation puts this rhythm at {p_auto:.0%} likely generated",
        ))

    # ---- keystroke rhythm ---------------------------------------------------
    if keys >= MIN_KEYS_FOR_RHYTHM:
        if int(f.get("dwell_samples") or 0) >= MIN_KEYS_FOR_RHYTHM and float(
            f.get("dwell_cv") or 0.0
        ) < ROBOTIC_DWELL_CV:
            hits.append((
                "keystroke:uniform_dwell", 0.82,
                f"Every key was held for almost exactly the same time (dwell variation "
                f"{float(f['dwell_cv']):.1%}); a human hand does not hold two keys for the same "
                "number of milliseconds, a timer does",
            ))
        if int(f.get("flight_samples") or 0) >= MIN_KEYS_FOR_RHYTHM and float(
            f.get("flight_cv") or 0.0
        ) < ROBOTIC_FLIGHT_CV:
            hits.append((
                "keystroke:uniform_flight", 0.70,
                f"The gaps between keystrokes are metronomic (variation "
                f"{float(f['flight_cv']):.1%}) rather than the burst-and-pause of typing",
            ))
        if float(f.get("typing_speed_cps") or 0.0) > IMPLAUSIBLE_TYPING_CPS:
            hits.append((
                "keystroke:implausible_speed", 0.66,
                f"Sustained {float(f['typing_speed_cps']):.1f} characters per second, "
                "above what a touch typist maintains on a form",
            ))
        if float(f.get("backspace_rate") or 0.0) == 0.0 and keys >= CONFIDENT_KEYS:
            hits.append((
                "keystroke:no_corrections", 0.28,
                f"{keys} keystrokes without a single correction - genuine fills of a "
                "long ID number almost always contain a backspace",
            ))

    flight_min = float(f.get("flight_min_ms") or 0.0)
    if keys >= 4 and 0.0 < flight_min < HUMAN_FLIGHT_FLOOR_MS:
        hits.append((
            "keystroke:below_human_floor", 0.93,
            f"Two keystrokes arrived {flight_min:.0f} ms apart, below the ~{HUMAN_FLIGHT_FLOOR_MS:.0f} ms "
            "floor a hand can physically achieve - these events were generated, not typed",
        ))

    # ---- pointer ------------------------------------------------------------
    pts = int(f.get("mouse_points") or 0)
    if pts == 0 and keys > 0:
        hits.append((
            "pointer:never_moved", 0.55,
            "The form was filled without the pointer moving once, which is what a "
            "scripted fill looks like from the browser's side",
        ))
    elif pts >= MIN_MOUSE_SAMPLES:
        if float(f.get("mouse_straightness") or 0.0) > STRAIGHT_LINE_RATIO:
            hits.append((
                "pointer:straight_line", 0.62,
                "The pointer travelled in a straight line between fields; a hand-driven "
                "cursor curves and overshoots",
            ))
        if int(f.get("mouse_dt_samples") or 0) >= 4 and float(
            f.get("mouse_dt_cv") or 0.0
        ) < ROBOTIC_MOUSE_DT_CV:
            hits.append((
                "pointer:uniform_sampling", 0.58,
                "Pointer samples arrived on a fixed interval - a synthesised path, not "
                "one the browser observed",
            ))
        if int(f.get("mouse_moving_segments") or 0) >= 4 and float(
            f.get("mouse_path_entropy") or 0.0
        ) < LOW_PATH_ENTROPY:
            hits.append((
                "pointer:no_tremor", 0.34,
                "The pointer path has almost no direction change, missing the tremor "
                "present in every human hand",
            ))

    # ---- form interaction ---------------------------------------------------
    if int(f.get("paste_into_sensitive") or 0) > 0:
        hits.append((
            "form:pasted_identity", 0.58,
            f"The identity number was pasted rather than typed - it was being read out "
            "of a file, not out of the applicant's own wallet",
        ))
    if float(f.get("field_order_monotonic") or 0.0) >= 1.0 and int(f.get("fields_touched") or 0) >= 4:
        hits.append((
            "form:robotic_order", 0.40,
            f"All {int(f['fields_touched'])} fields were filled strictly top to bottom with no "
            "field ever revisited, where a genuine applicant hesitates and scrolls back",
        ))

    # ---- timing -------------------------------------------------------------
    total = float(f.get("total_time_s") or 0.0)
    if 0.0 < total < MIN_PLAUSIBLE_FORM_SECONDS:
        hits.append((
            "timing:too_fast", 0.64,
            f"The whole form was completed in {total:.0f} s, too fast for the text it "
            "contains to have been read",
        ))
    if float(f.get("near_round_sleep_s") or 0.0) > 0.0:
        hits.append((
            "timing:hardcoded_sleep", 0.60,
            f"Time on form was {total:.1f} s, within a second of a round "
            f"{float(f['near_round_sleep_s']):.0f} s - the signature of a hardcoded sleep",
        ))

    # ---- environment --------------------------------------------------------
    flags = list(f.get("automation_flags") or [])
    if flags:
        hard = "navigator.webdriver" in flags or "automation user-agent" in flags
        hits.append((
            "environment:automation", 0.90 if hard else 0.35,
            "The browser identifies itself as automated (" + ", ".join(flags) + ")",
        ))
    if float(f.get("locale_incoherent") or 0.0) >= 1.0:
        hits.append((
            "environment:locale_mismatch", 0.45,
            "Device locale and clock disagree about where this submission came from - "
            + str(f.get("locale_detail") or ""),
        ))

    clean = 1.0
    for _, w, _r in hits:
        clean *= (1.0 - w)
    score = clamp(1.0 - clean)

    hard = any(w >= 0.90 for _, w, _r in hits)
    if not hard:
        score = min(score, 0.80)

    # Confidence is about how much we saw, not how bad it looked. Twelve
    # keystrokes and no pointer data cannot carry a rejection however damning
    # they are, and saying so is the difference between a signal and a guess.
    evidence = min(1.0, keys / CONFIDENT_KEYS) * 0.6 + min(1.0, pts / 60.0) * 0.25
    evidence += 0.15 if int(f.get("fields_touched") or 0) >= 3 else 0.0
    confidence = clamp(0.25 + 0.7 * evidence)
    if hard:
        # A webdriver flag or a sub-15 ms flight is self-evident; it does not need
        # a long sample to be worth acting on.
        confidence = max(confidence, 0.88)

    reasons = [r for _, _w, r in sorted(hits, key=lambda h: -h[1])]
    if not reasons:
        reasons.append(
            "Typing rhythm, pointer movement and time on form are all consistent with a "
            "human filling this out by hand"
        )
    return score, confidence, reasons, [{"rule": r, "weight": round(w, 3)} for r, w, _ in hits]


class BehavioralDetector(Detector):
    name = "behavioral"

    def applicable(self, payload: SubmissionPayload) -> bool:
        """Only run when the browser actually sent telemetry.

        Scoring an absent buffer as 0.0 would hand every fraudster a one-line
        bypass - omit the field, look human. Skipping instead pushes the decision
        to policy: a merchant that sets `require_behavioral` gets a REVIEW for
        any submission without telemetry, through the same 'required input
        missing' path that already covers liveness and ID documents.
        """
        feats = (payload.extra or {}).get("behavioral")
        return isinstance(feats, dict) and int(feats.get("event_count") or 0) > 0

    def _run(self, payload: SubmissionPayload) -> DetectorOutput:
        feats = dict((payload.extra or {}).get("behavioral") or {})
        score, confidence, reasons, hits = score_features(feats)

        signals: Dict[str, Any] = {
            "rule_hits": hits,
            "keystroke": {
                "count": feats.get("keystroke_count"),
                "dwell_mean_ms": feats.get("dwell_mean_ms"),
                "dwell_cv": feats.get("dwell_cv"),
                "dwell_samples": feats.get("dwell_samples"),
                "flight_mean_ms": feats.get("flight_mean_ms"),
                "flight_cv": feats.get("flight_cv"),
                "flight_min_ms": feats.get("flight_min_ms"),
                "typing_speed_cps": feats.get("typing_speed_cps"),
                "backspace_rate": feats.get("backspace_rate"),
            },
            "pointer": {
                "samples": feats.get("mouse_points"),
                "straightness": feats.get("mouse_straightness"),
                "path_entropy": feats.get("mouse_path_entropy"),
                "moving_segments": feats.get("mouse_moving_segments"),
                "sample_interval_cv": feats.get("mouse_dt_cv"),
                "jerk_cv": feats.get("mouse_jerk_cv"),
            },
            "form": {
                "fields_touched": feats.get("fields_touched"),
                "revisits": feats.get("field_revisits"),
                "filled_in_order": bool(feats.get("field_order_monotonic")),
                "paste_count": feats.get("paste_count"),
                "paste_into_sensitive": feats.get("paste_into_sensitive"),
                "total_time_s": feats.get("total_time_s"),
            },
            "environment": {
                "timezone": feats.get("timezone"),
                "timezone_offset_min": feats.get("timezone_offset_min"),
                "languages": feats.get("languages"),
                "automation_flags": feats.get("automation_flags"),
                "locale_incoherent": bool(feats.get("locale_incoherent")),
            },
            # The bands the rules above actually fire on, published rather than
            # duplicated in the dashboard. A UI that hard-codes "normal dwell is
            # 60-160 ms" drifts silently the moment a constant here is re-fitted,
            # and the reviewer is then reading a chart that no longer describes
            # the decision. Server owns the thresholds; the page only draws them.
            "reference": {
                "dwell_mean_ms": {"human": [55, 190], "measured": feats.get("dwell_mean_ms")},
                "dwell_cv": {"human": [ROBOTIC_DWELL_CV, 0.75], "measured": feats.get("dwell_cv")},
                "flight_cv": {"human": [ROBOTIC_FLIGHT_CV, 3.0], "measured": feats.get("flight_cv")},
                "flight_min_ms": {"human": [HUMAN_FLIGHT_FLOOR_MS, 400],
                                   "measured": feats.get("flight_min_ms")},
                "typing_speed_cps": {"human": [0.6, IMPLAUSIBLE_TYPING_CPS],
                                      "measured": feats.get("typing_speed_cps")},
                "rollover_rate": {"human": [0.02, 0.6], "measured": feats.get("rollover_rate")},
                "mouse_straightness": {"human": [0.0, STRAIGHT_LINE_RATIO],
                                        "measured": feats.get("mouse_straightness")},
                "mouse_path_entropy": {"human": [LOW_PATH_ENTROPY, 1.0],
                                        "measured": feats.get("mouse_path_entropy")},
                "mouse_dt_cv": {"human": [ROBOTIC_MOUSE_DT_CV, 2.0],
                                 "measured": feats.get("mouse_dt_cv")},
                "total_time_s": {"human": [MIN_PLAUSIBLE_FORM_SECONDS, 600],
                                  "measured": feats.get("total_time_s")},
            },
            "evidence": {
                "events_seen": feats.get("event_count"),
                "keystroke_model": (
                    None if model_p_automated(feats) is None
                    else round(float(model_p_automated(feats)), 4)
                ),
            },
        }
        return DetectorOutput(
            name=self.name, label=self.label, score=score, confidence=confidence,
            reasons=reasons, signals=signals,
        )
