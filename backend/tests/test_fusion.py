"""Fusion and policy behaviour — the logic that turns scores into decisions."""
import pytest

from verityne.config import MerchantPolicy
from verityne.fusion import decide, heuristic_score
from verityne.schemas import DetectorOutput


def det(name, score, confidence=0.8, status="ok"):
    return DetectorOutput(name=name, label=name, score=score, confidence=confidence, status=status)


def breakdown(**scores):
    return {n: det(n, s) for n, s in scores.items()}


class TestHeuristicScore:
    def test_all_clean_scores_low(self):
        b = breakdown(selfie_deepfake=0.05, id_forensics=0.05, face_match=0.05, metadata_exif=0.05)
        assert heuristic_score(b) < 0.2

    def test_one_loud_detector_is_not_buried(self):
        """Four quiet detectors must not average away one confident alarm —
        that is precisely the shape a fraudster optimises for."""
        b = breakdown(selfie_deepfake=0.05, id_forensics=0.97, face_match=0.05, metadata_exif=0.05)
        assert heuristic_score(b) > 0.4

    def test_low_confidence_detectors_carry_less_weight(self):
        confident = {"id_forensics": det("id_forensics", 0.9, confidence=0.95)}
        unsure = {"id_forensics": det("id_forensics", 0.9, confidence=0.05)}
        assert heuristic_score(confident) >= heuristic_score(unsure)

    def test_errored_detectors_are_ignored_not_counted_as_clean(self):
        b = {"id_forensics": det("id_forensics", 0.9), "face_match": det("face_match", 0.0, status="error")}
        assert heuristic_score(b) > 0.5

    def test_no_usable_detectors_returns_neutral(self):
        assert heuristic_score({"a": det("a", 0.0, status="error")}) == 0.5


class TestDecide:
    policy = MerchantPolicy(min_risk_for_review=0.4, min_risk_for_reject=0.75, abstain_band=0.05)

    def test_clear_pass(self):
        v, abstained = decide(0.10, self.policy)
        assert v == "PASS" and not abstained

    def test_clear_reject(self):
        v, abstained = decide(0.95, self.policy)
        assert v == "REJECT" and not abstained

    def test_abstains_just_above_the_reject_line(self):
        v, abstained = decide(0.77, self.policy)
        assert v == "REVIEW" and abstained

    def test_abstains_just_below_the_review_line(self):
        v, abstained = decide(0.38, self.policy)
        assert v == "REVIEW" and abstained

    def test_missing_required_liveness_blocks_a_pass(self):
        p = MerchantPolicy(require_liveness=True)
        b = {"liveness_video": det("liveness_video", 0.0, status="skipped")}
        v, abstained = decide(0.05, p, b)
        assert v == "REVIEW" and abstained

    def test_liveness_not_required_allows_a_pass(self):
        p = MerchantPolicy(require_liveness=False, require_id_document=False)
        b = {"liveness_video": det("liveness_video", 0.0, status="skipped")}
        assert decide(0.05, p, b)[0] == "PASS"

    def test_errored_detector_downgrades_a_pass_to_review(self):
        p = MerchantPolicy(require_liveness=False, require_id_document=False)
        b = {"selfie_deepfake": det("selfie_deepfake", 0.0, status="error")}
        v, abstained = decide(0.05, p, b)
        assert v == "REVIEW" and abstained

    def test_stricter_merchant_rejects_what_default_would_review(self):
        strict = MerchantPolicy(min_risk_for_review=0.25, min_risk_for_reject=0.60, abstain_band=0.0)
        assert decide(0.65, strict)[0] == "REJECT"
        assert decide(0.65, self.policy)[0] == "REVIEW"


class TestTrainedLayoutStaysPinned:
    """The learned fusion model is fitted on a *subset* of the detectors.

    Detector 6 is in `DETECTOR_NAMES` (it is a detector, and the dashboard shows
    it) but deliberately not in `FUSION_TRAINED_NAMES` (the corpus has no
    telemetry, so its column would be identically zero). Nothing in the type
    system stops the two lists drifting back together, and the failure is silent:
    the next `make train` would fit a model on a dead column and the ablation
    would report a detector as worthless when what is actually worthless is the
    data behind it.
    """

    def test_the_training_layout_excludes_behavioral(self):
        from verityne.config import DETECTOR_NAMES, FUSION_TRAINED_NAMES

        assert "behavioral" in DETECTOR_NAMES
        assert "behavioral" not in FUSION_TRAINED_NAMES
        assert set(FUSION_TRAINED_NAMES) < set(DETECTOR_NAMES)

    def test_the_training_script_and_the_serving_path_agree(self):
        """`train_fusion.build_matrix` and `fusion.training_feature_names` are two
        implementations of one layout, in two files, that must not diverge."""
        from train_fusion import build_matrix
        from verityne.fusion import training_feature_names

        _, _, names = build_matrix([])
        assert names == training_feature_names()

    def test_a_model_trained_before_detector_six_still_loads(self):
        """Features are looked up by name, so an older model asks for the columns
        it knows and never sees the one that was appended after it was fitted."""
        from verityne.fusion import feature_vector, training_feature_names

        b = breakdown(selfie_deepfake=0.8, liveness_video=0.3, id_forensics=0.6,
                      face_match=0.2, metadata_exif=0.5, behavioral=0.99)
        old_layout = training_feature_names()
        x, order = feature_vector(b, old_layout)
        assert order == old_layout
        assert len(x) == len(old_layout)
        assert not any(n.startswith("behavioral") for n in order)

    def test_a_missing_feature_is_an_error_not_a_silent_zero(self):
        from verityne.fusion import feature_vector

        with pytest.raises(ValueError, match="fusion features unavailable"):
            feature_vector(breakdown(selfie_deepfake=0.5), ["nonexistent_score"])
