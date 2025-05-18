import os
import glob
import redis
import shap
import joblib
import mlflow
import numpy as np
import pandas as pd
from datetime import datetime
from io import StringIO
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from evidently import Report
from evidently.presets import DataDriftPreset

# ───────────────────── Setup ─────────────────────
app = FastAPI()
mlflow.set_tracking_uri("http://localhost:5000")
model = mlflow.pyfunc.load_model("models:/xgb-fraud-model/Production")
input_schema = model.metadata.get_input_schema()
reference = pd.read_csv("reference.csv")
model_feature_columns = reference.columns.tolist()

r = redis.StrictRedis(host="localhost", port=6379, db=0, decode_responses=True)
LOG_DIR = "logged_batches"
os.makedirs(LOG_DIR, exist_ok=True)

# ───────────────────── Models ─────────────────────
class Transaction(BaseModel):
    transaction_id: str
    user_id: str
    amount: float
    merchant: str
    transaction_type: str
    timestamp: str
    location: str

# ───────────────────── Helpers ─────────────────────
def parse_isoformat_utc(ts):
    try:
        return datetime.strptime(ts, "%Y-%m-%dT%H:%M:%S.%fZ")
    except ValueError:
        return datetime.strptime(ts, "%Y-%m-%dT%H:%M:%SZ")


def extract_features(txn: dict) -> pd.DataFrame:
    user_key = f"user:{txn['user_id']}:features"
    try:
        amounts = list(map(float, r.lrange(f"{user_key}:amounts", 0, 4))) or []
        timestamps = r.lrange(f"{user_key}:timestamps", 0, 4)
        unique_merchants = r.scard(f"{user_key}:merchants") or 0
    except Exception:
        amounts, timestamps, unique_merchants = [], [], 0

    avg_txn_amt = np.mean(amounts) if amounts else txn['amount']
    var_txn_amt = np.var(amounts) if len(amounts) > 1 else 0
    now = parse_isoformat_utc(txn['timestamp'])
    txn_times = [parse_isoformat_utc(ts) for ts in timestamps if ts]
    txn_freq_1hr = sum(1 for t in txn_times if (now - t).total_seconds() <= 3600)

    try:
        last_txn_time = r.get(f"{user_key}:last_txn_time")
        time_since_last_txn = (now - parse_isoformat_utc(last_txn_time)).total_seconds() if last_txn_time else -1
    except:
        time_since_last_txn = -1

    # Start with raw features
    feat = {
        'amount': float(txn['amount']),
        'time_since_last_txn': float(time_since_last_txn),
        'avg_txn_amt': float(avg_txn_amt),
        'var_txn_amt': float(var_txn_amt),
        'unique_merchants': int(unique_merchants),
        'txn_freq_1hr': int(txn_freq_1hr),
    }

    # One-hot encodings as actual booleans
    for merchant_col in [c for c in model_feature_columns if c.startswith('merchant_')]:
        merchant_name = merchant_col.split('_', 1)[1]
        feat[merchant_col] = txn['merchant'] == merchant_name

    for tt_col in [c for c in model_feature_columns if c.startswith('transaction_type_')]:
        tt_name = tt_col.split('_', 2)[2]
        feat[tt_col] = txn['transaction_type'] == tt_name

    for loc_col in [c for c in model_feature_columns if c.startswith('location_')]:
        loc_name = loc_col.split('_', 1)[1]
        feat[loc_col] = txn['location'] == loc_name

    # Ensure all required columns exist
    for col in model_feature_columns:
        if col not in feat:
            feat[col] = False if col.startswith(('merchant_', 'transaction_type_', 'location_')) else 0.0

    # Build DataFrame and force types dynamically from model schema
    df = pd.DataFrame([feat])[model_feature_columns]

    for col in input_schema:
        if col.name in df.columns:
            if col.type == "double":
                df[col.name] = pd.to_numeric(df[col.name], errors="coerce").astype("float64")
            elif col.type == "long":
                df[col.name] = pd.to_numeric(df[col.name], errors="coerce").astype("int64")
            elif col.type == "boolean":
                df[col.name] = df[col.name].astype("bool")

    # Final debug print
    print("\n🧪 FINAL DF dtypes:")
    print(df.dtypes)
    print("\n🧪 FINAL DF sample row:")
    print(df.to_string(index=False))

    return df


# ───────────────────── Endpoints ─────────────────────

@app.get("/")
def root():
    return {"message": "Fraud detection API is live"}

@app.post("/predict")
def predict(txn: Transaction):
    txn_dict = txn.dict()
    try:
        features = extract_features(txn_dict)
        
        # Use raw predict
        risk = int(model.predict(features)[0])
        
        # Try probability only if available
        try:
            # Unwrap the native model (e.g., XGBClassifier)
            native_model = model.unwrap()
            prob = float(native_model.predict_proba(features)[0][1])
        except Exception:
            prob = 1.0 if risk == 1 else 0.0

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    try:
        timestamp = datetime.utcnow().strftime('%Y-%m-%d-%H')
        log_path = os.path.join(LOG_DIR, f"{timestamp}.csv")
        features["timestamp"] = datetime.utcnow()
        features.to_csv(log_path, mode='a', header=not os.path.exists(log_path), index=False)
    except Exception as e:
        print("⚠️ Logging failed:", e)

    return {
        "transaction_id": txn.transaction_id,
        "fraud_score": prob,
        "is_fraud": risk
    }


@app.post("/explain")
def explain(txn: Transaction):
    txn_dict = txn.dict()
    try:
        features = extract_features(txn_dict)
        explainer = shap.Explainer(model)
        shap_values = explainer(features)
        explanation = shap_values.values.tolist()[0]
        base_value = shap_values.base_values.tolist()[0]
        feature_names = features.columns.tolist()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return {
        "transaction_id": txn.transaction_id,
        "base_value": base_value,
        "shap_values": dict(zip(feature_names, explanation))
    }

@app.get("/drift-report", response_class=HTMLResponse)
def drift_report():
    reference_path = "reference.csv"
    if not os.path.exists(reference_path):
        return HTMLResponse(content="<h3>❌ Reference data not found</h3>", status_code=500)

    reference = pd.read_csv(reference_path)
    batch_files = sorted(glob.glob(os.path.join(LOG_DIR, "*.csv")))
    if not batch_files:
        return HTMLResponse(content="<h3>⚠️ No batch data logged yet</h3>", status_code=500)

    current = pd.read_csv(batch_files[-1])
    try:
        for col in reference.columns:
            if col not in current.columns:
                current[col] = 0
        current = current[reference.columns]

        report = Report([DataDriftPreset(method="psi")], include_tests=True)

        def sanitize_data(df):
            df = df.replace([np.inf, -np.inf], np.nan)
            df = df.fillna(1e-6)
            df = df.clip(lower=1e-6)
            return df.astype(float)

        reference = sanitize_data(reference)
        current = sanitize_data(current)

        result = report.run(reference, current)
        report_path = "drift_report.html"
        result.save_html(report_path)

        with open(report_path, "r", encoding="utf-8") as f:
            html_content = f.read()
        return HTMLResponse(content=html_content, status_code=200)

    except Exception as e:
        return HTMLResponse(
            content=f"<h3>❌ Failed to generate report</h3><pre>{str(e)}</pre>",
            status_code=500
        )
