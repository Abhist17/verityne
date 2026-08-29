"""Detector-level behaviour on synthesised inputs, plus the never-raise contract."""
import numpy as np
import pytest
from PIL import Image

from verityne.detectors.base import Detector, SubmissionPayload
from verityne.detectors.metadata_exif import MetadataExifDetector
from verityne.utils.ela import tamper_score
from verityne.utils.spectral import resample_residual, spectral_features


@pytest.fixture
def noise_image():
    rng = np.random.default_rng(0)
    return rng.integers(0, 255, (256, 256, 3), dtype=np.uint8)


class TestDetectorContract:
    def test_a_raising_detector_degrades_instead_of_propagating(self):
        """A broken detector must never take the whole verdict down with it."""

        class Exploding(Detector):
            name = "exploding"

            def _run(self, payload):
                raise RuntimeError("boom")

        out = Exploding().run(SubmissionPayload(submission_id="t"))
        assert out.status == "error"
        assert out.score == 0.0 and out.confidence == 0.0
        assert "boom" in (out.detail or "")

    def test_inapplicable_detector_is_skipped_not_scored(self):
        out = MetadataExifDetector().run(SubmissionPayload(submission_id="t"))
        assert out.status == "skipped"
        assert out.confidence == 0.0

    def test_scores_are_clamped_into_range(self):
        class OutOfRange(Detector):
            name = "wild"

            def _run(self, payload):
                from verityne.schemas import DetectorOutput

                return DetectorOutput(name="wild", label="wild", score=1.0, confidence=1.0)

        out = OutOfRange().run(SubmissionPayload(submission_id="t"))
        assert 0.0 <= out.score <= 1.0 and 0.0 <= out.confidence <= 1.0


class TestProvenanceSignal:
    def test_upscaled_image_has_a_lower_resample_residual(self):
        """The signal that catches an ID-card portrait passed off as a selfie."""
        rng = np.random.default_rng(1)
        sharp = rng.integers(0, 255, (512, 512), dtype=np.uint8).astype(np.float32)

        small = np.asarray(Image.fromarray(sharp.astype(np.uint8)).resize((96, 96), Image.LANCZOS))
        upscaled = np.asarray(Image.fromarray(small).resize((512, 512), Image.LANCZOS)).astype(np.float32)

        assert resample_residual(upscaled) < resample_residual(sharp)

    def test_spectral_features_are_finite_and_complete(self, noise_image):
        feats = spectral_features(noise_image)
        expected = {"hf_energy", "mf_energy", "ripple", "grid_peak", "resample_residual", "effective_scale"}
        assert expected.issubset(feats.keys())
        assert all(np.isfinite(v) for v in feats.values())


class TestEla:
    def test_uniform_image_is_not_flagged_as_tampered(self):
        flat = np.full((256, 256, 3), 128, dtype=np.uint8)
        assert tamper_score(flat)["score"] < 0.3

    def test_returns_a_wellformed_result(self, noise_image):
        r = tamper_score(noise_image)
        assert 0.0 <= r["score"] <= 1.0
        assert "regions" in r and isinstance(r["regions"], list)


class TestJsonSafety:
    """Numpy types in detector signals used to crash the request at the INSERT,
    long after the verdict had been computed correctly."""

    def test_numpy_scalars_are_converted(self):
        from verityne.utils.jsonsafe import to_jsonable

        out = to_jsonable({"i": np.int64(7), "f": np.float32(0.5), "b": np.bool_(True)})
        assert isinstance(out["i"], int) and isinstance(out["f"], float) and isinstance(out["b"], bool)

    def test_nested_and_array_values_are_converted(self):
        from verityne.utils.jsonsafe import to_jsonable

        out = to_jsonable({"regions": [{"bbox": np.array([1, 2, 3, 4]), "z": np.float64(2.5)}]})
        assert out["regions"][0]["bbox"] == [1, 2, 3, 4]
        assert isinstance(out["regions"][0]["z"], float)

    def test_non_finite_floats_become_null(self):
        from verityne.utils.jsonsafe import to_jsonable

        assert to_jsonable({"a": float("nan"), "b": np.float64("inf")}) == {"a": None, "b": None}

    def test_result_is_actually_serialisable(self):
        import json

        from verityne.utils.jsonsafe import to_jsonable

        payload = {"i": np.int64(3), "arr": np.arange(4), "nested": {"f": np.float32(1.5)}}
        json.dumps(to_jsonable(payload))  # must not raise
