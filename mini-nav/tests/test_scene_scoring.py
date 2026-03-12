"""Tests for scene scoring algorithm in multi-object retrieval."""

import pytest
from benchmarks.tasks.multi_object_retrieval import _compute_scene_score


class TestSceneScoringAlgorithm:
    """Test suite for scene scoring with co-occurrence penalty."""

    def test_scene_score_basic(self):
        """Test basic scene scoring with single match."""
        query_object_ids = ["obj_1", "obj_2", "obj_3"]

        # Scene A has obj_1
        retrieved_results = {
            "scene_A": [("distance_1", "obj_1")],
        }

        scores = _compute_scene_score(query_object_ids, retrieved_results, gamma=1.0)

        # Hit rate = 1/3, similarity = 1/(1+distance_1)
        assert "scene_A" in scores
        assert scores["scene_A"] > 0

    def test_scene_score_no_match(self):
        """Test scene scoring when no objects match."""
        query_object_ids = ["obj_1", "obj_2", "obj_3"]

        retrieved_results = {
            "scene_A": [("distance_1", "other_obj")],
        }

        scores = _compute_scene_score(query_object_ids, retrieved_results, gamma=1.0)

        assert scores["scene_A"] == 0.0

    def test_scene_score_multiple_scenes(self):
        """Test scoring across multiple scenes."""
        query_object_ids = ["obj_1", "obj_2"]

        retrieved_results = {
            "scene_A": [("0.1", "obj_1")],
            "scene_B": [("0.1", "obj_2")],
            "scene_C": [("0.1", "other")],
        }

        scores = _compute_scene_score(query_object_ids, retrieved_results, gamma=1.0)

        # Scenes with matches should have positive scores
        assert scores["scene_A"] > 0
        assert scores["scene_B"] > 0
        # Scene C has no match, score should be 0
        assert scores["scene_C"] == 0.0

    def test_scene_score_gamma_zero(self):
        """Test scoring with gamma=0 (no penalty)."""
        query_object_ids = ["obj_1", "obj_2", "obj_3", "obj_4", "obj_5"]

        retrieved_results = {
            "scene_A": [("0.1", "obj_1")],
        }

        scores_gamma_0 = _compute_scene_score(query_object_ids, retrieved_results, gamma=0.0)
        scores_gamma_1 = _compute_scene_score(query_object_ids, retrieved_results, gamma=1.0)

        # With gamma=0, hit_rate^0 = 1, so score = similarity
        # With gamma=1, hit_rate^1 = 1/5, so score = similarity * 1/5
        # scores_gamma_0 should be larger
        assert scores_gamma_0["scene_A"] > scores_gamma_1["scene_A"]

    def test_scene_score_multiple_matches(self):
        """Test scoring when scene has multiple matching objects."""
        query_object_ids = ["obj_1", "obj_2"]

        retrieved_results = {
            "scene_A": [("0.1", "obj_1"), ("0.2", "obj_2")],
        }

        scores = _compute_scene_score(query_object_ids, retrieved_results, gamma=1.0)

        # Both objects match, hit_rate = 2/2 = 1.0
        # Score = (1/(1+0.1) + 1/(1+0.2)) * 1.0
        expected_similarity = 1.0 / 1.1 + 1.0 / 1.2
        assert abs(scores["scene_A"] - expected_similarity) < 0.01

    def test_scene_score_distance_to_similarity(self):
        """Test that smaller distance yields higher score."""
        query_object_ids = ["obj_1"]

        retrieved_results = {
            "scene_close": [("0.01", "obj_1")],
            "scene_far": [("10.0", "obj_1")],
        }

        scores = _compute_scene_score(query_object_ids, retrieved_results, gamma=1.0)

        # Closer scene should have higher score
        assert scores["scene_close"] > scores["scene_far"]

    def test_scene_score_empty_results(self):
        """Test scoring with empty retrieved results."""
        query_object_ids = ["obj_1", "obj_2"]

        retrieved_results = {}

        scores = _compute_scene_score(query_object_ids, retrieved_results, gamma=1.0)

        assert scores == {}

    def test_scene_score_empty_query(self):
        """Test scoring with empty query objects."""
        query_object_ids = []

        retrieved_results = {
            "scene_A": [("0.1", "obj_1")],
        }

        scores = _compute_scene_score(query_object_ids, retrieved_results, gamma=1.0)

        # With empty query, no scenes should have positive score
        assert all(score == 0.0 for score in scores.values())
