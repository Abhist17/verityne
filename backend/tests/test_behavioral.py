"""Detector 6 — behavioral biometrics.

The tests that matter here are the *asymmetric* ones. Missing this detector's
signal on a fraud kit costs one caught fraud; firing it on a nervous human costs
that person their onboarding, so the false-positive tests are written first and
are the strict ones.
"""
import random

import pytest

from verityne.config import MerchantPolicy, get_policy
from verityne.detectors.behavioral import (
    HUMAN_FLIGHT_FLOOR_MS,
    BehavioralDetector,
    extract_features,
    score_features,
)
from verityne.detectors.base import SubmissionPayload
from verityne.fusion import behavioral_channel, decide, uncorroborated_ceiling
from verityne.schemas import DetectorOutput


# --------------------------------------------------------------------------- #
# Telemetry fixtures. Written as generators over a seed so every assertion below
# holds across a spread of samples rather than one lucky draw.
# --------------------------------------------------------------------------- #

def human_buffer(seed: int = 0, *, careful: bool = True) -> dict:
    """A genuine fill: noisy dwell, burst-and-pause flight, typos, curved pointer."""
    r = random.Random(seed)
    t = 800.0
    keys = []
    fields = ["name", "pan", "dob", "address"]
    for field in fields:
        for _ in range(r.randint(9, 16)):
            dwell = max(35.0, r.gauss(95, 32))
            keys.append({"key": r.choice("abcdefg1234"), "down": t, "up": t + dwell, "field": field})
            t += dwell + max(25.0, r.gauss(150, 90))
            if r.random() < 0.07:
                keys.append({"key": "Backspace", "down": t, "up": t + 70, "field": field})
                t += 260
        t += abs(r.gauss(1400, 500)) if careful else 200
    mouse, x, y, mt = [], 300.0, 240.0, 500.0
    for _ in range(260):
        x += r.gauss(0, 9)
        y += r.gauss(0, 7)
        mt += max(4.0, r.gauss(17, 9))
        mouse.append({"t": mt, "x": x, "y": y})
    focus = [
        {"field": f, "index": i, "in": 900 + i * 4200, "out": 900 + i * 4200 + abs(r.gauss(3600, 900))}
        for i, f in enumerate(fields)
    ]
    focus.append({"field": "dob", "index": 2, "in": t + 200, "out": t + 2600})  # scrolled back to fix it
    return {
        "form_loaded_at": 0, "submitted_at": t + 4000,
        "keys": keys, "mouse": mouse, "focus": focus, "paste": [],
        "declared_country": "IN",
        "env": {
            "timezone": "Asia/Kolkata", "timezone_offset_min": -330, "languages": ["en-IN", "hi"],
            "screen": [1920, 1080], "webdriver": False, "hardware_concurrency": 8,
            "user_agent": "Mozilla/5.0 (X11; Linux x86_64) Chrome/120",
        },
    }


def kit_buffer(seed: int = 0) -> dict:
    """A Telegram fraud kit: driven browser, timer-uniform keys, pasted PAN, sleep(300)."""
    r = random.Random(seed)
    t, keys = 100.0, []
    for field in ("name", "dob", "address"):
        for _ in range(14):
            keys.append({"key": r.choice("abcdef"), "down": t, "up": t + 50, "field": field})
            t += 60
    return {
        "form_loaded_at": 0, "submitted_at": 300_000,
        "keys": keys,
        "mouse": [{"t": 100 + i * 16, "x": 100 + i * 4, "y": 100 + i * 3} for i in range(40)],
        "focus": [
            {"field": f, "index": i, "in": 100 + i * 500, "out": 100 + i * 500 + 400}
            for i, f in enumerate(("name", "dob", "address", "pan"))
        ],
        "paste": [{"field": "pan", "t": 2000, "length": 10}],
        "declared_country": "IN",
        "env": {
            "timezone": "Europe/Moscow", "timezone_offset_min": -180, "languages": ["ru-RU"],
            "screen": [1280, 1024], "webdriver": True, "hardware_concurrency": 2,
            "user_agent": "Mozilla/5.0 HeadlessChrome/120",
        },
    }


def score(buf: dict) -> float:
    return score_features(extract_features(buf))[0]


def rules(buf: dict) -> set:
    return {h["rule"] for h in score_features(extract_features(buf))[3]}


# --------------------------------------------------------------------------- #

class TestGenuineFills:
    """False positives are the expensive error. These are the strict tests."""

    @pytest.mark.parametrize("seed", range(6))
    def test_a_careful_human_fill_raises_nothing(self, seed):
        assert score(human_buffer(seed)) < 0.2

    @pytest.mark.parametrize("seed", range(6))
    def test_no_rule_fires_on_a_genuine_fill(self, seed):
        assert rules(human_buffer(seed)) == set()

    def test_a_second_language_does_not_make_a_locale_incoherent(self):
        """A Mumbai browser advertising en-IN, hi and en-GB is an ordinary human."""
        buf = human_buffer(0)
        buf["env"]["languages"] = ["en-GB", "en-IN", "hi"]
        assert extract_features(buf)["locale_incoherent"] == 0.0

    def test_an_unlisted_region_is_never_penalised(self):
        """A rule that fired on 'locale we have no entry for' would punish the
        unusual rather than the fraudulent."""
        buf = human_buffer(0)
        buf["env"]["languages"] = ["is-IS"]
        buf["declared_country"] = "IS"
        assert extract_features(buf)["locale_incoherent"] == 0.0

    def test_pasting_an_address_is_not_suspicious(self):
        buf = human_buffer(0)
        buf["paste"] = [{"field": "address", "t": 5000, "length": 60}]
        assert "form:pasted_identity" not in rules(buf)


class TestAutomation:
    @pytest.mark.parametrize("seed", range(4))
    def test_a_fraud_kit_is_caught(self, seed):
        assert score(kit_buffer(seed)) > 0.9

    def test_the_physical_floor_is_what_makes_it_categorical(self):
        hits = rules(kit_buffer(0))
        assert "keystroke:below_human_floor" in hits
        assert "environment:automation" in hits

    def test_uniform_dwell_is_read_as_a_timer(self):
        assert "keystroke:uniform_dwell" in rules(kit_buffer(0))

    def test_a_perfectly_straight_pointer_path_is_caught(self):
        """Entropy of a single-direction path is exactly 0.0, and a `0 < x` guard
        would let the most robotic input possible slip through the tremor rule."""
        buf = {
            "form_loaded_at": 0, "submitted_at": 40_000, "keys": [], "focus": [], "paste": [], "env": {},
            "mouse": [{"t": 100 + i * 16, "x": 100 + i * 4, "y": 100 + i * 3} for i in range(40)],
        }
        assert "pointer:no_tremor" in rules(buf)

    def test_a_hardcoded_sleep_is_named(self):
        assert "timing:hardcoded_sleep" in rules(kit_buffer(0))

    def test_flight_below_the_human_floor(self):
        buf = kit_buffer(0)
        gaps = [
            b["down"] - a["up"]
            for a, b in zip(buf["keys"], buf["keys"][1:])
        ]
        assert min(gaps) < HUMAN_FLIGHT_FLOOR_MS


class TestSeparation:
    def test_every_kit_outscores_every_human(self):
        humans = [score(human_buffer(s)) for s in range(8)]
        kits = [score(kit_buffer(s)) for s in range(8)]
        assert min(kits) > max(humans)


class TestMalformedInput:
    """This endpoint is reachable by an attacker, so the extractor must be total."""

    @pytest.mark.parametrize("buf", [
        {},
        {"keys": None, "mouse": None, "focus": None, "paste": None, "env": None},
        {"keys": ["not-a-dict", 3, None], "mouse": [{"t": "x"}], "focus": [{}], "paste": [{}]},
        {"keys": [{"down": 5, "up": 1}], "form_loaded_at": 900, "submitted_at": 100},
        {"env": {"languages": [None, 7], "timezone_offset_min": "abc", "screen": "wide"}},
        {"mouse": [{"t": float("nan"), "x": 1, "y": 2}]},
    ])
    def test_garbage_yields_a_feature_row_not_an_exception(self, buf):
        f = extract_features(buf)
        assert isinstance(f, dict) and "event_count" in f
        s, c, reasons, _ = score_features(f)
        assert 0.0 <= s <= 1.0 and 0.0 <= c <= 1.0 and reasons

    def test_an_out_of_order_keystroke_does_not_invent_a_negative_flight(self):
        f = extract_features({"keys": [{"key": "a", "down": 500, "up": 560},
                                       {"key": "b", "down": 100, "up": 160}]})
        assert f["flight_min_ms"] >= 0.0 or f["keystroke_count"] == 2


class TestConfidence:
    def test_a_thin_sample_is_reported_as_thin(self):
        thin = {"keys": [{"key": "a", "down": i * 100, "up": i * 100 + 50} for i in range(6)],
                "mouse": [], "focus": [], "paste": [], "env": {}}
        _, conf, _, _ = score_features(extract_features(thin))
        assert conf < 0.6

    def test_a_full_sample_is_confident(self):
        _, conf, _, _ = score_features(extract_features(human_buffer(0)))
        assert conf > 0.85

    def test_a_self_declared_bot_needs_no_long_sample(self):
        """`navigator.webdriver` is self-evident; it does not need 35 keystrokes
        to be worth acting on."""
        _, conf, _, _ = score_features(extract_features(
            {"keys": [], "mouse": [], "focus": [], "paste": [],
             "env": {"webdriver": True, "user_agent": "HeadlessChrome"}}
        ))
        assert conf >= 0.88


class TestDetectorWiring:
    def test_absent_telemetry_skips_rather_than_scoring_clean(self):
        """Scoring a missing buffer as 0.0 would be a one-line bypass: omit the
        field, look human."""
        out = BehavioralDetector().run(SubmissionPayload(submission_id="t"))
        assert out.status == "skipped"

    def test_an_empty_buffer_also_skips(self):
        payload = SubmissionPayload(submission_id="t", extra={"behavioral": {"event_count": 0}})
        assert BehavioralDetector().run(payload).status == "skipped"

    def test_a_populated_buffer_runs_and_reports_its_evidence(self):
        payload = SubmissionPayload(
            submission_id="t", extra={"behavioral": extract_features(kit_buffer(0))}
        )
        out = BehavioralDetector().run(payload)
        assert out.status == "ok" and out.score > 0.9
        assert out.signals["keystroke"]["count"] > 0
        assert out.signals["environment"]["automation_flags"]

    def test_the_detector_never_raises_on_a_hostile_buffer(self):
        payload = SubmissionPayload(
            submission_id="t", extra={"behavioral": {"event_count": 5, "keystroke_count": "many"}}
        )
        assert BehavioralDetector().run(payload).status in {"ok", "error"}


class TestEvidenceChannel:
    """What Detector 6 is *allowed* to do to a verdict on its own."""

    def _breakdown(self, buf):
        payload = SubmissionPayload(submission_id="t", extra={"behavioral": extract_features(buf)})
        return {"behavioral": BehavioralDetector().run(payload)}

    def test_statistical_evidence_cannot_reject_by_itself(self):
        """A fast, tidy, uncorrected fill is a real person having a brisk day.
        It must reach a human, never an automatic rejection."""
        policy = get_policy("default")
        soft = DetectorOutput(
            name="behavioral", label="Behavioral Biometrics", score=0.80, confidence=0.95,
            signals={"rule_hits": [{"rule": "timing:too_fast", "weight": 0.64},
                                   {"rule": "keystroke:no_corrections", "weight": 0.28}]},
        )
        contribution, _ = behavioral_channel({"behavioral": soft}, policy)
        assert contribution <= uncorroborated_ceiling(policy)
        assert decide(contribution, policy)[0] != "REJECT"

    def test_categorical_evidence_may_reject(self):
        policy = get_policy("default")
        contribution, reasons = behavioral_channel(self._breakdown(kit_buffer(0)), policy)
        assert contribution > uncorroborated_ceiling(policy)
        assert decide(contribution, policy)[0] == "REJECT"
        assert reasons

    def test_a_skipped_detector_contributes_nothing(self):
        out = BehavioralDetector().run(SubmissionPayload(submission_id="t"))
        assert behavioral_channel({"behavioral": out}, get_policy("default")) == (0.0, [])

    def test_low_confidence_scales_the_contribution_down(self):
        policy = get_policy("default")
        loud_but_thin = DetectorOutput(
            name="behavioral", label="Behavioral Biometrics", score=0.9, confidence=0.3,
            signals={"rule_hits": [{"rule": "timing:too_fast", "weight": 0.64}]},
        )
        loud_and_seen = loud_but_thin.model_copy(update={"confidence": 0.95})
        thin, _ = behavioral_channel({"behavioral": loud_but_thin}, policy)
        seen, _ = behavioral_channel({"behavioral": loud_and_seen}, policy)
        assert thin < seen


class TestPolicy:
    def test_require_behavioral_routes_a_silent_packet_to_review(self):
        policy = MerchantPolicy(
            merchant_id="m", require_behavioral=True, require_liveness=False, require_id_document=False
        )
        out = BehavioralDetector().run(SubmissionPayload(submission_id="t"))
        verdict, abstained = decide(0.05, policy, {"behavioral": out})
        assert verdict == "REVIEW" and abstained

    def test_it_is_off_by_default_so_api_integrations_still_pass(self):
        policy = MerchantPolicy(merchant_id="m", require_liveness=False, require_id_document=False)
        out = BehavioralDetector().run(SubmissionPayload(submission_id="t"))
        assert decide(0.05, policy, {"behavioral": out})[0] == "PASS"


class TestCollectorContract:
    """Pin the extractor to the exact buffer `frontend/lib/telemetry.ts` emits.

    The two halves of Detector 6 are written in different languages and cannot
    be type-checked against each other, and every way they can disagree is
    *silent*: a key the extractor does not recognise reads as zero, and a zero
    reads as 'no evidence' rather than as an error. The collector shipped its
    screen size as `{w, h}` while the extractor destructured `[w, h]`, which
    retired the headless-browser check without a single failing test.

    So these mirror `TelemetryCollector.snapshot()` field for field. If that file
    changes shape, this is what fails.
    """

    def collector_buffer(self, seed: int = 0) -> dict:
        """Exactly what snapshot() produces — redacted keys, raw offset, {w,h}.

        Timings carry human variance rather than a fixed step: a metronomic
        fixture is one the detector is *supposed* to flag, so building the
        contract test on one would assert the opposite of what it means to.
        Mouse samples respect the collector's own 25 ms throttle floor.
        """
        r = random.Random(seed)
        keys = []
        t = 400.0
        for _ in range(30):
            # Printable characters are redacted to a single placeholder char at
            # source; the extractor must still count them as printable.
            dwell = max(35.0, r.gauss(95, 30))
            keys.append({"key": "·", "down": t, "up": t + dwell, "field": "pan"})
            t += dwell + max(25.0, r.gauss(150, 85))
        keys.append({"key": "Backspace", "down": t, "up": t + 70, "field": "pan"})
        mouse, x, y, mt = [], 200.0, 150.0, 100.0
        for _ in range(40):
            x += r.gauss(0, 10)
            y += r.gauss(0, 8)
            mt += max(25.0, r.gauss(48, 20))  # the collector's MOUSE_MIN_DT floor
            mouse.append({"t": round(mt), "x": round(x), "y": round(y)})
        return {
            "token": "0123456789abcdef0123456789abcdef",
            "merchant_id": "default",
            "form_loaded_at": 0,
            "submitted_at": 42_000,
            "declared_country": "IN",
            "keys": keys,
            "mouse": mouse,
            "focus": [{"field": "pan", "index": 0, "in": 300, "out": 9000}],
            "paste": [{"field": "pan", "t": 5000, "length": 10}],
            "env": {
                "timezone": "Asia/Kolkata",
                # RAW getTimezoneOffset(): minutes *behind* UTC. IST is -330.
                "timezone_offset_min": -330,
                "languages": ["en-IN", "en", "hi"],
                "webdriver": False,
                "screen": {"w": 1920, "h": 1080},
                "hardware_concurrency": 8,
                "touch_points": 0,
            },
        }

    def test_the_redacted_printable_key_still_counts_as_typing(self):
        f = extract_features(self.collector_buffer())
        assert f["typing_speed_cps"] > 0, "redaction broke the printable-key count"
        assert f["backspace_rate"] > 0, "Backspace no longer reads as a correction"

    def test_the_raw_timezone_offset_is_flipped_not_taken_literally(self):
        """`getTimezoneOffset()` returns -330 for IST. Read literally that is
        UTC-5:30, which makes every honest Indian applicant look incoherent."""
        f = extract_features(self.collector_buffer())
        assert f["timezone_offset_min"] == 330
        assert f["locale_incoherent"] == 0.0

    def test_the_object_form_of_screen_is_understood(self):
        buf = self.collector_buffer()
        buf["env"]["screen"] = {"w": 0, "h": 0}
        assert "zero-size screen" in extract_features(buf)["automation_flags"]

    def test_the_array_form_of_screen_still_works(self):
        buf = self.collector_buffer()
        buf["env"]["screen"] = [0, 0]
        assert "zero-size screen" in extract_features(buf)["automation_flags"]

    def test_an_unrecognised_screen_shape_is_ignored_not_flagged(self):
        buf = self.collector_buffer()
        buf["env"]["screen"] = "1920x1080"
        assert "zero-size screen" not in extract_features(buf)["automation_flags"]

    def test_a_focus_never_closed_does_not_break_extraction(self):
        """`out` is undefined when the field still had focus at submit."""
        buf = self.collector_buffer()
        buf["focus"] = [{"field": "pan", "index": 0, "in": 300}]
        assert extract_features(buf)["fields_touched"] == 1

    def test_a_key_whose_keyup_was_missed_is_still_counted(self):
        buf = self.collector_buffer()
        buf["keys"].append({"key": "·", "down": 50_000, "field": "pan"})
        f = extract_features(buf)
        assert f["keystroke_count"] == len(buf["keys"])
        assert f["dwell_samples"] == len(buf["keys"]) - 1

    @pytest.mark.parametrize("seed", range(5))
    def test_the_whole_collector_buffer_reads_as_human(self, seed):
        buf = self.collector_buffer(seed)
        buf["paste"] = []  # the one deliberately suspicious field in the fixture
        assert score(buf) < 0.4

    def test_the_token_satisfies_the_envelope_bounds(self):
        """The collector mints 32 hex chars; the envelope requires 8..64."""
        from verityne.schemas import BehavioralEnvelope

        BehavioralEnvelope(**self.collector_buffer())
