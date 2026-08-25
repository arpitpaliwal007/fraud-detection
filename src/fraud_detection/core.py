from __future__ import annotations

import math
import random


FEATURES = ["log_amount", "hour_risk", "distance", "velocity_1h", "new_device", "foreign", "merchant_risk"]


def _sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-value)
        return 1 / (1 + z)
    z = math.exp(value)
    return z / (1 + z)


def generate_transactions(count: int = 12000, seed: int = 31) -> list[dict]:
    rng = random.Random(seed)
    rows = []
    for idx in range(count):
        amount = min(rng.lognormvariate(3.25, 1.0), 5000)
        hour = rng.randrange(24)
        distance = rng.expovariate(1 / 24)
        velocity = min(int(rng.expovariate(1 / 1.1)), 12)
        new_device = int(rng.random() < 0.12)
        foreign = int(rng.random() < 0.07)
        merchant_risk = rng.betavariate(1.4, 7)
        # Pattern drift: device-based fraud becomes more prevalent in the final quarter.
        drift = 0.9 * new_device if idx > count * 0.75 else 0.0
        logit = (-6.0 + 0.52 * math.log1p(amount) + 0.85 * (hour < 5) + 0.016 * distance
                 + 0.42 * velocity + 1.05 * new_device + 1.25 * foreign + 3.1 * merchant_risk + drift)
        label = int(rng.random() < _sigmoid(logit))
        rows.append({"timestamp": idx, "amount": round(amount, 2), "log_amount": math.log1p(amount),
                     "hour_risk": int(hour < 5), "distance": distance, "velocity_1h": velocity,
                     "new_device": new_device, "foreign": foreign, "merchant_risk": merchant_risk,
                     "fraud": label})
    return rows


class LogisticFraudModel:
    def __init__(self, learning_rate: float = 0.08, epochs: int = 280, positive_weight: float = 7.0):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.positive_weight = positive_weight
        self.means: list[float] = []
        self.scales: list[float] = []
        self.weights: list[float] = [0.0] * (len(FEATURES) + 1)

    def _vector(self, row: dict) -> list[float]:
        return [1.0] + [(float(row[f]) - m) / s for f, m, s in zip(FEATURES, self.means, self.scales)]

    def fit(self, rows: list[dict]) -> "LogisticFraudModel":
        columns = [[float(row[f]) for row in rows] for f in FEATURES]
        self.means = [sum(col) / len(col) for col in columns]
        self.scales = [max((sum((v - m) ** 2 for v in col) / len(col)) ** 0.5, 1e-8)
                       for col, m in zip(columns, self.means)]
        x = [self._vector(row) for row in rows]
        y = [row["fraud"] for row in rows]
        for epoch in range(self.epochs):
            gradient = [0.0] * len(self.weights)
            for vector, target in zip(x, y):
                prediction = _sigmoid(sum(w * value for w, value in zip(self.weights, vector)))
                sample_weight = self.positive_weight if target else 1.0
                for j, value in enumerate(vector):
                    gradient[j] += sample_weight * (prediction - target) * value
            decay = self.learning_rate / math.sqrt(1 + epoch * 0.04)
            for j in range(len(self.weights)):
                regularization = 0.002 * self.weights[j] if j else 0.0
                self.weights[j] -= decay * (gradient[j] / len(rows) + regularization)
        return self

    def predict_proba(self, row: dict) -> float:
        vector = self._vector(row)
        return _sigmoid(sum(w * value for w, value in zip(self.weights, vector)))

    def explain(self, row: dict, top_k: int = 4) -> list[dict]:
        vector = self._vector(row)
        contributions = [(feature, self.weights[i + 1] * vector[i + 1]) for i, feature in enumerate(FEATURES)]
        contributions.sort(key=lambda pair: abs(pair[1]), reverse=True)
        return [{"feature": feature, "log_odds_contribution": round(value, 4),
                 "effect": "increases risk" if value > 0 else "decreases risk"}
                for feature, value in contributions[:top_k]]

    def to_dict(self) -> dict:
        return {"features": FEATURES, "means": self.means, "scales": self.scales, "weights": self.weights,
                "positive_weight": self.positive_weight}


def classification_at_threshold(rows: list[dict], scores: list[float], threshold: float) -> dict:
    tp = fp = tn = fn = 0
    recovered = total_fraud_amount = 0.0
    for row, score in zip(rows, scores):
        pred, actual = score >= threshold, bool(row["fraud"])
        tp += pred and actual
        fp += pred and not actual
        tn += not pred and not actual
        fn += not pred and actual
        if actual:
            total_fraud_amount += row["amount"]
            if pred:
                recovered += row["amount"]
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    return {"precision": precision, "recall": recall, "false_positive_rate": fp / max(fp + tn, 1),
            "amount_recall": recovered / max(total_fraud_amount, 1), "expected_cost": fn * 500 + fp * 8,
            "alerts": tp + fp, "tp": tp, "fp": fp, "fn": fn}


def choose_threshold(rows: list[dict], scores: list[float], min_precision: float = 0.55) -> float:
    candidates = sorted(set(scores))
    best = None
    for threshold in candidates:
        result = classification_at_threshold(rows, scores, threshold)
        if result["precision"] >= min_precision and (best is None or result["recall"] > best[0]):
            best = (result["recall"], threshold)
    if best:
        return best[1]
    return min(candidates, key=lambda t: classification_at_threshold(rows, scores, t)["expected_cost"])


def _auc(rows: list[dict], scores: list[float], kind: str) -> float:
    pairs = sorted(zip(scores, (r["fraud"] for r in rows)), reverse=True)
    positives = sum(label for _, label in pairs)
    negatives = len(pairs) - positives
    tp = fp = 0
    points = [(0.0, 1.0 if kind == "pr" else 0.0)]
    for _, label in pairs:
        tp += label
        fp += 1 - label
        recall = tp / max(positives, 1)
        ordinate = tp / max(tp + fp, 1) if kind == "pr" else fp / max(negatives, 1)
        points.append((recall if kind == "pr" else ordinate, ordinate if kind == "pr" else recall))
    area = 0.0
    for (x1, y1), (x2, y2) in zip(points, points[1:]):
        area += (x2 - x1) * (y1 + y2) / 2
    return area


def evaluate(rows: list[dict], scores: list[float], threshold: float) -> dict:
    result = classification_at_threshold(rows, scores, threshold)
    result.update({"threshold": threshold, "pr_auc": _auc(rows, scores, "pr"), "roc_auc": _auc(rows, scores, "roc")})
    return {key: round(value, 5) if isinstance(value, float) else value for key, value in result.items()}

