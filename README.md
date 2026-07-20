# Toll Gate ML Prototype

Two models trained on toll-gate transaction data:

1. **Fraud detection** — classifies each transaction as fraudulent or legitimate
   (tampered amounts, impossible speeds, duplicate scans, axle/weight mismatches).
2. **Traffic & revenue forecasting** — predicts next-hour vehicle volume and
   revenue per plaza, beating a naive "same hour yesterday" baseline by ~60%.

No real toll fraud dataset is publicly available, so this repo ships a
**synthetic data generator** that mimics a real toll network schema
(plaza/lane/vehicle class/payment method/axle count/weight/speed). Swap
`src/generate_data.py` for a loader against your real database export
once you're ready — the model scripts only care about final column names.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## 1. Generate data

```bash
python src/generate_data.py --days 90 --out data/toll_data.csv
```

`--days 90` generates roughly 2.5M rows (10 plazas x realistic hourly
traffic). Use a smaller value like `--days 14` for a quick local test run.

**To use your real data instead:** export it to CSV with these columns and
skip this step: `vehicle_id, timestamp, plaza_id, lane, vehicle_class,
payment_method, axle_count, weight_kg, amount, speed_kmph,
inter_arrival_sec, fraud_label, fraud_reason`. If you don't have
`fraud_label` yet (most real systems don't), see the note at the bottom.

## 2. Train the fraud model

```bash
python src/fraud_model.py --data data/toll_data.csv --out models/fraud_model.joblib
```

Prints AUC, precision/recall, confusion matrix, and top feature importances,
then saves the trained model + feature column list to `models/`.

Note: this is tuned for high **recall** (catch as much fraud/revenue loss as
possible) using `class_weight="balanced"`, which trades off precision (more
false positives for staff to review). If false positives are too costly for
your ops team, either drop `class_weight="balanced"` in
`src/fraud_model.py` or raise the decision threshold on
`clf.predict_proba()` output instead of using the default `.predict()`.

## 3. Train the forecasting models

```bash
python src/forecast_model.py --data data/toll_data.csv --out-prefix models/forecast
```

Trains and saves two models (`models/forecast_volume.joblib`,
`models/forecast_revenue.joblib`) that predict next-hour vehicle count and
revenue per plaza using lag (1h/24h/168h) and rolling-average features.

## Loading a saved model later

```python
import joblib
bundle = joblib.load("models/fraud_model.joblib")
model, feature_columns = bundle["model"], bundle["feature_columns"]

# Make sure your inference-time features are built the same way as in
# fraud_model.build_features(), then:
proba = model.predict_proba(X[feature_columns])[:, 1]
```

## If you don't have fraud labels in real data

Most toll operators don't have pre-labeled fraud. Two practical paths:

1. **Rule-seeded labels**: flag known-bad patterns (negative/zero amounts,
   physically impossible speeds, duplicate scans under N seconds) as a
   starting label set, same as this generator does. This gets you a working
   v1 model fast.
2. **Unsupervised anomaly detection**: swap `RandomForestClassifier` for
   `sklearn.ensemble.IsolationForest` on the same features, no labels
   required — flags outliers for manual review, and you use the reviewed
   outcomes to build a labeled dataset for a supervised model like this one
   over time.

## Project structure

```
toll-ml-prototype/
├── data/                  # generated/real CSVs (gitignored)
├── models/                # saved .joblib models (gitignored)
├── notebooks/             # optional exploration space
├── src/
│   ├── generate_data.py   # synthetic data generator
│   ├── fraud_model.py     # fraud classifier training
│   └── forecast_model.py  # volume/revenue forecasting training
├── requirements.txt
└── README.md
```
