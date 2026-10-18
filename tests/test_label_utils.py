"""Unit tests for Task 5.1 & 5.2 — label mapping and IoU utilities."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from src.data.label_utils import cvb_behavior_to_label, cbvd5_actions_to_label, bbox_iou, match_predicted_to_gt


# ── §3.2 CVB behavior string → label ID ──────────────────────────────────────

class TestCvbBehaviorToLabel:
    def test_resting_standing(self):
        assert cvb_behavior_to_label("resting-standing") == 0

    def test_resting_lying(self):
        assert cvb_behavior_to_label("resting-lying") == 1

    def test_grazing(self):
        assert cvb_behavior_to_label("grazing") == 2

    def test_drinking(self):
        assert cvb_behavior_to_label("drinking") == 3

    def test_ruminating_standing(self):
        assert cvb_behavior_to_label("ruminating-standing") == 4

    def test_ruminating_lying(self):
        assert cvb_behavior_to_label("ruminating-lying") == 4

    def test_grooming(self):
        assert cvb_behavior_to_label("grooming") == 5

    def test_other(self):
        assert cvb_behavior_to_label("other") == 6

    # SKIP behaviors
    def test_skip_hidden(self):
        assert cvb_behavior_to_label("hidden") is None

    def test_skip_walking(self):
        assert cvb_behavior_to_label("walking") is None

    def test_skip_running(self):
        assert cvb_behavior_to_label("running") is None

    def test_skip_none(self):
        assert cvb_behavior_to_label("none") is None

    def test_unknown_returns_none(self):
        assert cvb_behavior_to_label("unknown_behavior") is None

    def test_case_insensitive(self):
        assert cvb_behavior_to_label("Grazing") == 2
        assert cvb_behavior_to_label("DRINKING") == 3

    def test_strips_whitespace(self):
        assert cvb_behavior_to_label("  grooming  ") == 5


# ── §3.3 CBVD-5 action ID → label ID (single label) ─────────────────────────

class TestCbvd5SingleAction:
    def test_stand(self):
        assert cbvd5_actions_to_label([1]) == 0

    def test_lying_down(self):
        assert cbvd5_actions_to_label([2]) == 1

    def test_foraging(self):
        assert cbvd5_actions_to_label([3]) == 2

    def test_drinking_water(self):
        assert cbvd5_actions_to_label([4]) == 3

    def test_rumination(self):
        assert cbvd5_actions_to_label([5]) == 4


# ── §3.4 Priority rule for multi-label CBVD-5 ────────────────────────────────

class TestCbvd5PriorityRule:
    def test_standing_foraging_picks_foraging(self):
        # (1,3) → Foraging (label 2)
        assert cbvd5_actions_to_label([1, 3]) == 2

    def test_lying_ruminating_picks_ruminating(self):
        # (2,5) → Ruminating (label 4)
        assert cbvd5_actions_to_label([2, 5]) == 4

    def test_standing_ruminating_picks_ruminating(self):
        # (1,5) → Ruminating (label 4)
        assert cbvd5_actions_to_label([1, 5]) == 4

    def test_standing_drinking_picks_drinking(self):
        # (1,4) → Drinking (label 3)
        assert cbvd5_actions_to_label([1, 4]) == 3

    def test_drinking_beats_foraging(self):
        assert cbvd5_actions_to_label([3, 4]) == 3

    def test_drinking_beats_all(self):
        assert cbvd5_actions_to_label([1, 2, 3, 4, 5]) == 3

    def test_foraging_beats_ruminating(self):
        assert cbvd5_actions_to_label([3, 5]) == 2

    def test_ruminating_beats_lying(self):
        assert cbvd5_actions_to_label([2, 5]) == 4

    def test_lying_beats_standing(self):
        assert cbvd5_actions_to_label([1, 2]) == 1

    def test_unknown_id_ignored(self):
        # action 99 unknown, falls through to [1] → Standing
        assert cbvd5_actions_to_label([1, 99]) == 0

    def test_all_unknown_raises(self):
        with pytest.raises(ValueError):
            cbvd5_actions_to_label([99, 100])


# ── Task 5.2: bbox_iou ────────────────────────────────────────────────────────

class TestBboxIou:
    def test_identical_boxes(self):
        box = [10, 10, 50, 50]
        assert bbox_iou(box, box) == pytest.approx(1.0)

    def test_zero_overlap(self):
        assert bbox_iou([0, 0, 10, 10], [20, 20, 30, 30]) == pytest.approx(0.0)

    def test_touching_edges_no_overlap(self):
        # share an edge but no area overlap
        assert bbox_iou([0, 0, 10, 10], [10, 0, 20, 10]) == pytest.approx(0.0)

    def test_half_overlap(self):
        # box_a covers [0,0,10,10]=100; box_b covers [5,0,15,10]=100; inter=[5,0,10,10]=50
        iou = bbox_iou([0, 0, 10, 10], [5, 0, 15, 10])
        assert iou == pytest.approx(50 / 150)

    def test_contained_box(self):
        # inner fully inside outer: inter=inner area
        outer = [0, 0, 10, 10]  # area 100
        inner = [2, 2, 8, 8]    # area 36
        iou = bbox_iou(outer, inner)
        assert iou == pytest.approx(36 / 100)

    def test_symmetry(self):
        a = [0, 0, 10, 20]
        b = [5, 5, 15, 25]
        assert bbox_iou(a, b) == pytest.approx(bbox_iou(b, a))


# ── Task 5.2: match_predicted_to_gt ──────────────────────────────────────────

class TestMatchPredictedToGt:
    def test_perfect_match_single(self):
        box = [[10, 10, 50, 50]]
        result = match_predicted_to_gt(box, box)
        assert result == {0: 0}

    def test_perfect_match_multiple(self):
        preds = [[0, 0, 10, 10], [20, 20, 30, 30]]
        gts   = [[0, 0, 10, 10], [20, 20, 30, 30]]
        result = match_predicted_to_gt(preds, gts)
        assert result == {0: 0, 1: 1}

    def test_zero_overlap_returns_empty(self):
        preds = [[0, 0, 5, 5]]
        gts   = [[100, 100, 110, 110]]
        result = match_predicted_to_gt(preds, gts)
        assert result == {}

    def test_below_threshold_excluded(self):
        # IoU ~0.14 < 0.3 threshold
        preds = [[0, 0, 10, 10]]
        gts   = [[8, 0, 18, 10]]  # inter=[8,0,10,10]=20; union=180; iou≈0.11
        result = match_predicted_to_gt(preds, gts, iou_threshold=0.3)
        assert result == {}

    def test_empty_pred_list(self):
        assert match_predicted_to_gt([], [[0, 0, 10, 10]]) == {}

    def test_empty_gt_list(self):
        assert match_predicted_to_gt([[0, 0, 10, 10]], []) == {}

    def test_one_to_one_not_many_to_one(self):
        # two identical preds vs one GT — only one gets matched
        preds = [[0, 0, 10, 10], [0, 0, 10, 10]]
        gts   = [[0, 0, 10, 10]]
        result = match_predicted_to_gt(preds, gts)
        assert len(result) == 1
        assert list(result.values()) == [0]

    def test_cross_match_avoids_duplicate_gt(self):
        # gt[0] overlaps pred[0] and pred[1] equally; Hungarian assigns optimally
        preds = [[0, 0, 10, 10], [0, 0, 10, 10]]
        gts   = [[0, 0, 10, 10], [20, 20, 30, 30]]
        result = match_predicted_to_gt(preds, gts)
        # gt indices must be unique (one-to-one)
        assert len(set(result.values())) == len(result)
