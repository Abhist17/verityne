"""The deterministic parts of the real-data ingest.

Same rule as the rest of the suite: assert on the arithmetic and the parsing,
never on a model's output. A regression in how LFW folds are read, or in how a
document quad is ordered, would silently corrupt a headline number, and that is
exactly the kind of thing a test should catch. Whether FaceNet scores 0.98 on
LFW belongs in `eval/face_match_lfw.json`, not in a red test.
"""

import numpy as np

from build_real_docs import _ink_ratio, _overlaps, pick_compatible
from calibrate_face_match_lfw import best_threshold, parse_pairs, rate_at_far, score_threshold
from evaluate_real_docs import hits_truth, iou
from midv2020 import _order_quad, _regions
from real_video import Clip, balance, detect_layout


# ---------------------------------------------------------------------------------
# LFW protocol
# ---------------------------------------------------------------------------------

class TestLfwPairs:
    def _write(self, tmp_path, folds=2, per_class=2):
        """A miniature pairs.txt in the exact format LFW ships."""
        lines = [f"{folds}\t{per_class}"]
        for f in range(folds):
            for i in range(per_class):
                lines.append(f"Person{f}{i}\t1\t2")
            for i in range(per_class):
                lines.append(f"AaA{f}{i}\t1\tBbB{f}{i}\t3")
        p = tmp_path / "pairs.txt"
        p.write_text("\n".join(lines))
        return p

    def test_labels_and_folds_follow_the_file(self, tmp_path):
        pairs, labels, folds = parse_pairs(self._write(tmp_path), tmp_path / "img")
        assert len(pairs) == 8
        # Two folds, each 2 matched then 2 mismatched.
        assert labels.tolist() == [1, 1, 0, 0, 1, 1, 0, 0]
        assert folds.tolist() == [0, 0, 0, 0, 1, 1, 1, 1]

    def test_filenames_are_zero_padded_to_four_digits(self, tmp_path):
        pairs, _, _ = parse_pairs(self._write(tmp_path), tmp_path / "img")
        a, b = pairs[0]
        assert a.name == "Person00_0001.jpg"
        assert b.name == "Person00_0002.jpg"

    def test_mismatched_pairs_name_two_different_people(self, tmp_path):
        pairs, labels, _ = parse_pairs(self._write(tmp_path), tmp_path / "img")
        a, b = pairs[2]  # first mismatched row
        assert labels[2] == 0
        assert a.parent.name != b.parent.name


class TestThresholds:
    def test_best_threshold_separates_a_clean_split(self):
        sims = np.array([0.1, 0.2, 0.8, 0.9], dtype=np.float32)
        labels = np.array([0, 0, 1, 1])
        thr, acc = best_threshold(sims, labels)
        assert acc == 1.0
        assert 0.2 < thr < 0.8

    def test_score_threshold_reports_tar_and_far(self):
        sims = np.array([0.1, 0.6, 0.7, 0.9], dtype=np.float32)
        labels = np.array([0, 0, 1, 1])
        r = score_threshold(sims, labels, 0.55)
        assert r["tar"] == 1.0          # both genuine pairs (0.7, 0.9) clear 0.55
        assert r["far"] == 0.5          # one of two impostors (0.6) also clears it
        assert r["accuracy"] == 0.75

    def test_far_budget_under_one_impostor_cannot_accept_any(self):
        # With 10 impostors, a 0.1% budget is less than one whole pair, so the
        # threshold must sit above every impostor rather than admitting one.
        sims = np.concatenate([np.linspace(0.0, 0.5, 10), np.linspace(0.6, 0.9, 10)])
        labels = np.array([0] * 10 + [1] * 10)
        r = rate_at_far(sims, labels, 0.001)
        assert r["far"] == 0.0
        # The reported threshold must itself deliver the reported FAR: applying
        # it with `>=` may not readmit the top impostor it was set to exclude.
        assert r["threshold"] >= 0.5
        assert float((sims[labels == 0] >= r["threshold"]).mean()) == 0.0

    def test_far_1pct_admits_roughly_that_share(self):
        rng = np.random.default_rng(0)
        sims = np.concatenate([rng.normal(0.0, 0.1, 1000), rng.normal(0.8, 0.1, 1000)])
        labels = np.array([0] * 1000 + [1] * 1000)
        r = rate_at_far(sims, labels, 0.01)
        assert r["far"] <= 0.011


# ---------------------------------------------------------------------------------
# MIDV-2020
# ---------------------------------------------------------------------------------

class TestMidvAnnotations:
    def test_quad_is_ordered_clockwise_from_top_left(self):
        # Deliberately scrambled input; the corners must come back tl, tr, br, bl.
        quad = np.array([[10, 90], [90, 10], [10, 10], [90, 90]], dtype=np.float32)
        out = _order_quad(quad)
        assert out.tolist() == [[10, 10], [90, 10], [90, 90], [10, 90]]

    def test_perspective_quad_orders_by_corner_geometry(self):
        quad = np.array([[12, 14], [200, 6], [210, 120], [4, 130]], dtype=np.float32)
        out = _order_quad(quad)
        assert out[0].tolist() == [12, 14]    # top-left: smallest x+y
        assert out[2].tolist() == [210, 120]  # bottom-right: largest x+y

    def test_regions_extracts_face_rect_and_doc_quad(self):
        entry = {
            "regions": [
                {"shape_attributes": {"name": "rect", "x": 10, "y": 20, "width": 30, "height": 40},
                 "region_attributes": {"field_name": "face"}},
                {"shape_attributes": {"name": "polygon", "all_points_x": [0, 100, 100, 0],
                                      "all_points_y": [0, 0, 60, 60]},
                 "region_attributes": {"field_name": "doc_quad"}},
            ]
        }
        face, quad = _regions(entry)
        assert face == (10, 20, 40, 60)  # x1, y1, x2, y2
        assert quad.shape == (4, 2)

    def test_missing_regions_degrade_to_none(self):
        assert _regions({"regions": []}) == (None, None)


# ---------------------------------------------------------------------------------
# Tamper construction
# ---------------------------------------------------------------------------------

class TestTamperPlacement:
    def test_overlap_is_symmetric_and_excludes_touching_edges(self):
        assert _overlaps((0, 0, 10, 10), (5, 5, 15, 15))
        assert not _overlaps((0, 0, 10, 10), (10, 0, 20, 10))  # share an edge only

    def test_pick_compatible_rejects_wildly_different_shapes(self):
        import random
        dst = [(0, 0, 200, 20)]           # a long thin line
        src = [(0, 0, 20, 200)]           # a tall thin column
        assert pick_compatible(dst, src, random.Random(0)) is None

    def test_pick_compatible_accepts_similar_shapes(self):
        import random
        dst = [(0, 0, 200, 20)]
        src = [(0, 0, 210, 22)]
        pair = pick_compatible(dst, src, random.Random(0))
        assert pair is not None
        assert pair[0] == (0, 0, 200, 20)

    def test_copy_move_never_selects_a_box_as_its_own_source(self):
        import random
        boxes = [(0, 0, 100, 20), (0, 40, 104, 21)]
        for seed in range(10):
            pair = pick_compatible(boxes, boxes, random.Random(seed), exclude_identical=True)
            if pair is not None:
                assert pair[0] != pair[1]

    def test_ink_ratio_is_high_for_text_and_low_for_flat_colour(self):
        flat = np.full((40, 200), 200, dtype=np.uint8)
        assert _ink_ratio(flat) == 0.0

        text = np.full((40, 200), 230, dtype=np.uint8)
        text[15:25, 20:180] = 30  # a dark bar standing in for a printed line
        assert _ink_ratio(text) > 0.1


# ---------------------------------------------------------------------------------
# Localisation scoring
# ---------------------------------------------------------------------------------

class TestLocalisation:
    def test_iou_of_identical_boxes_is_one(self):
        assert iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0

    def test_iou_of_disjoint_boxes_is_zero(self):
        assert iou((0, 0, 10, 10), (50, 50, 60, 60)) == 0.0

    def test_a_flagged_region_covering_the_edit_counts_as_a_hit(self):
        regions = [{"bbox": [0, 0, 100, 100]}]
        hit, best = hits_truth(regions, (10, 10, 50, 50))
        assert hit
        assert best > 0

    def test_a_flagged_region_elsewhere_is_not_a_hit(self):
        regions = [{"bbox": [200, 200, 260, 260]}]
        hit, best = hits_truth(regions, (10, 10, 50, 50))
        assert not hit
        assert best == 0.0

    def test_only_the_top_k_regions_are_considered(self):
        # The overlapping region sits fourth, past the top-3 cut.
        regions = [{"bbox": [300, 300, 310, 310]}] * 3 + [{"bbox": [0, 0, 100, 100]}]
        hit, _ = hits_truth(regions, (10, 10, 50, 50), top_k=3)
        assert not hit


# ---------------------------------------------------------------------------------
# Real video ingest
# ---------------------------------------------------------------------------------

class TestRealVideo:
    def test_layout_detection(self, tmp_path):
        assert detect_layout(tmp_path) is None

        (tmp_path / "ffpp" / "original_sequences").mkdir(parents=True)
        assert detect_layout(tmp_path / "ffpp") == "faceforensics"

        (tmp_path / "cdf" / "Celeb-synthesis").mkdir(parents=True)
        assert detect_layout(tmp_path / "cdf") == "celebdf"

        (tmp_path / "dfdc").mkdir()
        (tmp_path / "dfdc" / "dataset.json").write_text("{}")
        assert detect_layout(tmp_path / "dfdc") == "dfdc"

    def test_balance_caps_each_class_and_each_method(self, tmp_path):
        clips = [Clip(tmp_path / f"r{i}.mp4", 0, "real", "ffpp") for i in range(50)]
        clips += [Clip(tmp_path / f"d{i}.mp4", 1, "Deepfakes", "ffpp") for i in range(50)]
        clips += [Clip(tmp_path / f"n{i}.mp4", 1, "NeuralTextures", "ffpp") for i in range(50)]

        out = balance(clips, per_class=10)
        assert sum(1 for c in out if c.method == "real") == 10
        # 10 across two methods: five each, so no method dominates the fake class.
        assert sum(1 for c in out if c.method == "Deepfakes") == 5
        assert sum(1 for c in out if c.method == "NeuralTextures") == 5

    def test_balance_is_deterministic_for_a_seed(self, tmp_path):
        clips = [Clip(tmp_path / f"r{i}.mp4", 0, "real", "ffpp") for i in range(20)]
        a = [c.path for c in balance(clips, 5, seed=3)]
        b = [c.path for c in balance(clips, 5, seed=3)]
        assert a == b
