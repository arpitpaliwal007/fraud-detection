# Real-Time Fraud Detection

An explainable, cost-aware transaction scoring system built around the constraints that
matter in fraud operations: extreme class imbalance, chronological drift, alert capacity,
probability calibration, and asymmetric business cost.

## Highlights

- Reproducible transaction stream with evolving fraud patterns
- Chronological train/validation/test split
- Standardized logistic model trained with fraud-weighted loss
- Threshold selection at a minimum precision requirement
- PR-AUC, ROC-AUC, recall, precision, expected cost, and amount-recovery metrics
- Per-transaction additive feature explanations
- FastAPI endpoint, tests, Docker, CI, and JSON model artifacts

## Quickstart

```bash
python scripts/run_demo.py
python -m unittest discover -s tests
```

Optional API:

```bash
pip install -e '.[api]'
uvicorn fraud_detection.api:app --reload --port 8002
```

## Evaluation design

Transactions are sorted by time. The first 65% trains the model, the next 15% selects
the operating threshold, and the final 20% is untouched test data. This exposes temporal
degradation that a random split would conceal. PR-AUC is the principal ranking metric;
the final decision metric is expected operational cost.

## Architecture

```text
transaction -> validation -> feature standardization -> probability + explanation
                                                      -> threshold -> approve/review
```

## Production extensions

Replace the in-process features with a point-in-time-correct feature store, add delayed
label reconciliation, shadow deployment, drift alerts, analyst feedback, and champion/
challenger model routing.

