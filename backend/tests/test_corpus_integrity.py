"""Guards against the corpus writing its own labels into the artefacts.

A generator setting that only ever fires on one class is not a feature, it is
the answer key. A detector that reads it scores perfectly and has learned
nothing, and every number computed downstream is inflated by an amount nobody
can see from the aggregate.

This is not hypothetical here. `pick_exif_mode` shipped for the life of the
project drawing `stale`, `edited` and `generated` for fraudulent packets only.
`metadata_exif` read that key into a 0.874 AUC and half the fused model's
above-chance performance before `ablate_fusion.py` caught it - see
`eval/ablation.json`. These tests exist so that it cannot come back quietly.
"""

import random

from ablate_fusion import leak_audit
from build_dataset import PHYSICALLY_FRAUD_ONLY, audit_exif_balance, pick_exif_mode


def entry(label: str, selfie_mode: str, id_mode: str | None = None) -> dict:
    return {
        "label": label,
        "selfie_exif": {"exif_mode": selfie_mode},
        "id_exif": {"exif_mode": id_mode or selfie_mode},
    }


# ---------------------------------------------------------------------------------
# The generator itself
# ---------------------------------------------------------------------------------

class TestPickExifMode:
    def _draw(self, n=4000, seed=0):
        """Draw both classes over every attack type, as build_packet does."""
        rng = random.Random(seed)
        out = []
        attacks = [None, "generated_selfie", "tampered_document", "impersonation",
                   "reused_id_selfie", "invalid_document", "face_swap_liveness"]
        for i in range(n):
            is_fake = i % 2 == 1
            attack = attacks[i % len(attacks)] if is_fake else None
            out.append(("fake" if is_fake else "real", pick_exif_mode(is_fake, attack, rng)))
        return out

    def test_no_mode_is_drawn_for_one_class_only(self):
        """The regression. Every mode a camera can produce must reach both classes."""
        draws = self._draw()
        modes = {m for _, m in draws}
        for mode in modes - set(PHYSICALLY_FRAUD_ONLY):
            labels = {label for label, m in draws if m == mode}
            assert labels == {"real", "fake"}, f"{mode} was drawn for {labels} only"

    def test_stale_and_edited_reach_genuine_packets(self):
        """Named explicitly, because these two are the ones that leaked.

        Both describe things that happen to honest people constantly: a photo
        picked out of the gallery, a selfie cropped before upload.
        """
        genuine = {m for label, m in self._draw() if label == "real"}
        assert "stale" in genuine
        assert "edited" in genuine

    def test_fraud_stays_enriched_so_the_detector_keeps_a_signal(self):
        """Balanced is not the same as uninformative.

        A reused or repurposed asset really is likelier to be stale or edited.
        The corpus should keep that tilt - it just cannot be the whole label.
        """
        draws = self._draw()
        for mode in ("stale", "edited"):
            fake = sum(1 for label, m in draws if m == mode and label == "fake")
            real = sum(1 for label, m in draws if m == mode and label == "real")
            assert fake > real, f"{mode} should stay enriched in fraud"
            assert real / (real + fake) > 0.15, f"{mode} is too rare on genuine packets to be honest"

    def test_the_stale_media_attack_still_always_looks_stale(self):
        rng = random.Random(1)
        modes = {pick_exif_mode(True, "stale_or_edited_media", rng) for _ in range(200)}
        assert modes <= {"stale", "edited", "generated"}

    def test_the_allowlist_carries_a_justification(self):
        """A mode is exempt only with a stated physical reason, not by fiat."""
        assert PHYSICALLY_FRAUD_ONLY
        for mode, why in PHYSICALLY_FRAUD_ONLY.items():
            assert isinstance(why, str) and len(why) > 20


# ---------------------------------------------------------------------------------
# The build-time guard
# ---------------------------------------------------------------------------------

class TestAuditExifBalance:
    def test_a_balanced_corpus_passes(self):
        entries = [entry("real", "fresh") for _ in range(20)] + [entry("fake", "fresh") for _ in range(20)]
        assert audit_exif_balance(entries) == []

    def test_a_fraud_only_mode_is_reported(self):
        entries = [entry("real", "fresh") for _ in range(20)] + [entry("fake", "stale") for _ in range(20)]
        complaints = audit_exif_balance(entries)
        assert any("stale" in c for c in complaints)

    def test_a_genuine_only_mode_is_reported_too(self):
        """The leak is one-sidedness, not the direction of it."""
        entries = [entry("real", "plain") for _ in range(20)] + [entry("fake", "fresh") for _ in range(20)]
        assert any("plain" in c for c in audit_exif_balance(entries))

    def test_the_allowlisted_mode_does_not_trip_it(self):
        # `fresh` has to sit on both classes here, or it is the leak under test.
        entries = [entry("real", "fresh") for _ in range(20)] + \
                  [entry("fake", "fresh") for _ in range(20)] + \
                  [entry("fake", "generated") for _ in range(20)]
        assert audit_exif_balance(entries) == []

    def test_a_handful_of_packets_is_not_called_a_leak(self):
        """Three one-sided rows is sampling noise; 26 is a rule."""
        entries = [entry("real", "fresh") for _ in range(20)] + \
                  [entry("fake", "fresh") for _ in range(20)] + \
                  [entry("fake", "stale") for _ in range(3)]
        assert audit_exif_balance(entries) == []

    def test_it_checks_the_id_document_too_not_only_the_selfie(self):
        entries = [entry("real", "fresh", "fresh") for _ in range(20)] + \
                  [entry("fake", "fresh", "edited") for _ in range(20)]
        assert any("id_exif" in c for c in audit_exif_balance(entries))

    def test_a_real_corpus_shaped_draw_is_clean(self):
        """End to end: what pick_exif_mode actually produces must pass the guard."""
        rng = random.Random(5)
        entries = []
        for i in range(600):
            is_fake = i % 2 == 1
            attack = "stale_or_edited_media" if is_fake and i % 12 == 1 else None
            entries.append(entry("fake" if is_fake else "real",
                                 pick_exif_mode(is_fake, attack, rng),
                                 pick_exif_mode(is_fake, attack, rng)))
        assert audit_exif_balance(entries) == []


# ---------------------------------------------------------------------------------
# The reporting audit
# ---------------------------------------------------------------------------------

class TestLeakAudit:
    def test_it_names_the_one_sided_value_and_counts_the_packets(self):
        entries = [entry("real", "fresh") for _ in range(30)] + [entry("fake", "stale") for _ in range(30)]
        report = leak_audit(entries)["selfie_exif_mode"]
        assert [l["value"] for l in report["one_sided_values"]] == ["fresh", "stale"]
        assert report["packets_carrying_a_one_sided_value"] == 60

    def test_p_fraud_is_reported_per_value(self):
        entries = [entry("real", "fresh") for _ in range(30)] + \
                  [entry("fake", "fresh") for _ in range(10)]
        values = leak_audit(entries)["selfie_exif_mode"]["values"]
        assert values["fresh"] == {"genuine": 30, "fraud": 10, "p_fraud": 0.25}

    def test_a_well_mixed_corpus_reports_nothing(self):
        entries = [entry("real", m) for m in ["fresh", "stale"] * 15] + \
                  [entry("fake", m) for m in ["fresh", "stale"] * 15]
        assert leak_audit(entries)["selfie_exif_mode"]["one_sided_values"] == []

    def test_an_allowlisted_value_is_reported_but_marked_expected(self):
        """Reported, so a reader can check the reasoning; not a defect.

        The distinction is what lets `make ablate` fail a build on a real leak
        without failing on the one value that is one-sided because cameras are.
        """
        entries = [entry("real", "fresh") for _ in range(20)] + \
                  [entry("fake", "fresh") for _ in range(20)] + \
                  [entry("fake", "generated") for _ in range(20)]
        found = leak_audit(entries)["selfie_exif_mode"]["one_sided_values"]
        assert [l["value"] for l in found] == ["generated"]
        assert found[0]["expected"] is True
        assert found[0]["justification"]

    def test_an_unjustified_one_sided_value_is_not_marked_expected(self):
        entries = [entry("real", "fresh") for _ in range(20)] + \
                  [entry("fake", "fresh") for _ in range(20)] + \
                  [entry("fake", "stale") for _ in range(20)]
        found = leak_audit(entries)["selfie_exif_mode"]["one_sided_values"]
        assert [(l["value"], l["expected"]) for l in found] == [("stale", False)]

    def test_the_two_audits_agree_on_what_is_allowed(self):
        """`audit_exif_balance` gates the build, `leak_audit` gates the report.

        They must not disagree about which values are exempt, or one of them
        will pass a corpus the other calls broken.
        """
        entries = [entry("real", "fresh") for _ in range(20)] + \
                  [entry("fake", "fresh") for _ in range(20)] + \
                  [entry("fake", "generated") for _ in range(20)]
        assert audit_exif_balance(entries) == []
        assert all(l["expected"] for l in leak_audit(entries)["selfie_exif_mode"]["one_sided_values"])


# ---------------------------------------------------------------------------------
# The face-match band's lower bound has a better source than this corpus
# ---------------------------------------------------------------------------------

class TestPreserveBetterLowBound:
    """`make pipeline` used to silently undo `make calibrate-face`.

    The corpus can fit the band's upper bound - LFW has no identity documents, so
    there is nothing better. It cannot fit the lower one: its genuine pairs derive
    from a single photograph each and sit near cosine 0.94, so fitting on them
    produced ~0.83, a threshold LFW showed rejects nearly half of real genuine
    pairs. Re-running calibration must not reinstate that.
    """

    CORPUS = {
        "low": 0.8321,
        "high": 0.9859,
        "fitted_on": "99 genuine / 21 mismatch / 11 duplicate pairs (train split)",
    }
    LFW = {
        "low": 0.4065,
        "high": 0.9878,
        "same_person": 0.4065,
        "far_1pct_threshold": 0.48,
        "far_0.1pct_threshold": 0.5198,
        "low_previously": 0.7835,
        "fitted_on": "low: LFW 10-fold, 6000 real pairs, accuracy 0.9820±0.0040; high: corpus",
        "lfw": {"accuracy": 0.982, "n_pairs": 6000},
    }

    def test_an_lfw_fitted_low_bound_survives_recalibration(self):
        from calibrate import preserve_better_low_bound

        merged = preserve_better_low_bound(dict(self.CORPUS), self.LFW)
        assert merged["low"] == 0.4065
        assert merged["same_person"] == 0.4065
        assert merged["lfw"]["n_pairs"] == 6000

    def test_the_corpus_upper_bound_is_still_taken(self):
        """The high bound is corpus-derived on purpose and must keep updating."""
        from calibrate import preserve_better_low_bound

        merged = preserve_better_low_bound(dict(self.CORPUS), self.LFW)
        assert merged["high"] == self.CORPUS["high"]

    def test_the_corpus_value_is_recorded_not_discarded(self):
        from calibrate import preserve_better_low_bound

        merged = preserve_better_low_bound(dict(self.CORPUS), self.LFW)
        assert merged["corpus_fitted_low"] == 0.8321
        assert "train split" in merged["corpus_fitted_on"]

    def test_provenance_still_describes_the_bound_actually_in_force(self):
        from calibrate import preserve_better_low_bound

        merged = preserve_better_low_bound(dict(self.CORPUS), self.LFW)
        assert "LFW" in merged["fitted_on"]

    def test_a_first_run_with_no_lfw_calibration_uses_the_corpus(self):
        from calibrate import preserve_better_low_bound

        assert preserve_better_low_bound(dict(self.CORPUS), {})["low"] == 0.8321

    def test_a_band_without_the_lfw_key_is_not_treated_as_better(self):
        """Only a band carrying LFW's own report counts as better-sourced."""
        from calibrate import preserve_better_low_bound

        stale = {"low": 0.9, "high": 0.99, "fitted_on": "some earlier corpus run"}
        assert preserve_better_low_bound(dict(self.CORPUS), stale)["low"] == 0.8321
