import unittest

from recommender import (
    MOVIES, UserProfile, all_tags, parse_interests, recommend, weighted_jaccard,
)


class SimilarityTests(unittest.TestCase):
    def test_identical_sets_score_one(self):
        self.assertEqual(weighted_jaccard({"a": 1.0, "b": 1.0}, {"a", "b"}), 1.0)

    def test_no_overlap_scores_zero(self):
        self.assertEqual(weighted_jaccard({"a": 1.0}, {"b"}), 0.0)

    def test_plain_jaccard_value(self):
        # shared = 1, union = 3  ->  1/3
        self.assertAlmostEqual(weighted_jaccard({"a": 1.0, "b": 1.0}, {"b", "c"}), 1 / 3)

    def test_empty_profile(self):
        self.assertEqual(weighted_jaccard({}, {"a"}), 0.0)


class InputTests(unittest.TestCase):
    def setUp(self):
        self.tags = all_tags()

    def test_cleans_case_and_spaces(self):
        valid, unknown = parse_interests("  SCI-FI ,Drama ", self.tags)
        self.assertEqual(valid, ["sci-fi", "drama"])
        self.assertEqual(unknown, [])

    def test_fixes_small_typo(self):
        valid, _ = parse_interests("romence", self.tags)
        self.assertEqual(valid, ["romance"])

    def test_unknown_and_duplicates(self):
        valid, unknown = parse_interests("horror, horror, pizza", self.tags)
        self.assertEqual(valid, ["horror"])
        self.assertEqual(unknown, ["pizza"])

    def test_empty_input(self):
        self.assertEqual(parse_interests("", self.tags), ([], []))


class RecommendTests(unittest.TestCase):
    def test_best_match_is_relevant(self):
        recs = recommend(UserProfile(["romance", "drama"]))
        self.assertIn(recs[0][0], {"The Notebook", "La La Land", "Pride and Prejudice"})

    def test_results_sorted_and_limited(self):
        recs = recommend(UserProfile(["action", "thriller"]), top_n=4)
        self.assertLessEqual(len(recs), 4)
        scores = [r[1] for r in recs]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_cold_start_falls_back_to_popular(self):
        recs = recommend(UserProfile([]), top_n=3)
        self.assertEqual(len(recs), 3)
        self.assertTrue(all(r[1] == 0.0 for r in recs))

    def test_rated_movies_are_excluded(self):
        p = UserProfile(["sci-fi"])
        p.rate("Inception", 5)
        self.assertNotIn("Inception", [r[0] for r in recommend(p, top_n=30)])

    def test_high_rating_boosts_tag_weight(self):
        p = UserProfile(["sci-fi"])
        before = p.weights["sci-fi"]
        p.rate("Interstellar", 5)
        self.assertGreater(p.weights["sci-fi"], before)

    def test_low_rating_penalises_disliked_tags(self):
        p = UserProfile(["action"])
        p.rate("The Notebook", 1)          # romance, drama -> disliked
        self.assertIn("romance", p.disliked)
        self.assertEqual(p.score(MOVIES["La La Land"]["tags"]), 0.0)

    def test_invalid_rating_rejected(self):
        with self.assertRaises(ValueError):
            UserProfile([]).rate("Inception", 9)


if __name__ == "__main__":
    unittest.main()
