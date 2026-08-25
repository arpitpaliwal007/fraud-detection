import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fraud_detection.core import LogisticFraudModel, choose_threshold, evaluate, generate_transactions


class FraudTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = generate_transactions(1200)
        cls.model = LogisticFraudModel(epochs=80).fit(cls.rows[:800])

    def test_probability_is_bounded(self):
        scores = [self.model.predict_proba(row) for row in self.rows[800:850]]
        self.assertTrue(all(0 <= score <= 1 for score in scores))

    def test_explanations_are_sorted(self):
        explanation = self.model.explain(self.rows[-1])
        magnitudes = [abs(item["log_odds_contribution"]) for item in explanation]
        self.assertEqual(magnitudes, sorted(magnitudes, reverse=True))

    def test_threshold_and_metrics(self):
        validation = self.rows[800:]
        scores = [self.model.predict_proba(row) for row in validation]
        threshold = choose_threshold(validation, scores, min_precision=0.3)
        report = evaluate(validation, scores, threshold)
        self.assertTrue(math.isfinite(threshold))
        self.assertTrue(0 <= report["pr_auc"] <= 1)
        self.assertTrue(0 <= report["roc_auc"] <= 1)


if __name__ == "__main__":
    unittest.main()

