from __future__ import annotations

import json
import time
from pathlib import Path

from .core import LogisticFraudModel, choose_threshold, evaluate, generate_transactions


def run_pipeline(output_dir: str | Path = "artifacts") -> dict:
    rows = generate_transactions()
    train_end, validation_end = int(len(rows) * 0.65), int(len(rows) * 0.80)
    train, validation, test = rows[:train_end], rows[train_end:validation_end], rows[validation_end:]
    model = LogisticFraudModel().fit(train)
    validation_scores = [model.predict_proba(row) for row in validation]
    threshold = choose_threshold(validation, validation_scores)
    start = time.perf_counter()
    test_scores = [model.predict_proba(row) for row in test]
    latency_ms = (time.perf_counter() - start) * 1000 / len(test)
    report = evaluate(test, test_scores, threshold)
    report.update({"fraud_rate": round(sum(r["fraud"] for r in test) / len(test), 5),
                   "mean_inference_ms": round(latency_ms, 6), "train_rows": len(train), "test_rows": len(test)})
    high_risk = sorted(zip(test, test_scores), key=lambda pair: pair[1], reverse=True)[:5]
    examples = [{"transaction": row, "score": round(score, 5), "explanation": model.explain(row)}
                for row, score in high_risk]
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    (target / "model.json").write_text(json.dumps({**model.to_dict(), "threshold": threshold}, indent=2))
    (target / "metrics.json").write_text(json.dumps(report, indent=2))
    (target / "explanations.json").write_text(json.dumps(examples, indent=2))
    return report

