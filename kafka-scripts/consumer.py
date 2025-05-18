from confluent_kafka import Consumer
import redis
import json
import mlflow.pyfunc
from datetime import datetime
import numpy as np
import pandas as pd

# Kafka config
conf = {
    'bootstrap.servers': 'localhost:9092',
    'group.id': 'fraud-detector',
    'auto.offset.reset': 'earliest'
}
consumer = Consumer(conf)
consumer.subscribe(['transactions'])

# Redis config for feature storage
r = redis.StrictRedis(host='localhost', port=6379, db=0, decode_responses=True)

# MLflow model
mlflow.set_tracking_uri("http://localhost:5000")
model = mlflow.xgboost.load_model("models:/xgb-fraud-model/Production")

def parse_isoformat_utc(ts):
    try:
        return datetime.strptime(ts, "%Y-%m-%dT%H:%M:%S.%fZ")
    except ValueError:
        return datetime.strptime(ts, "%Y-%m-%dT%H:%M:%SZ")

def update_user_features(txn):
    user_key = f"user:{txn['user_id']}:features"
    pipe = r.pipeline()
    pipe.lpush(f"{user_key}:amounts", txn['amount'])
    pipe.ltrim(f"{user_key}:amounts", 0, 4)
    pipe.lpush(f"{user_key}:timestamps", txn['timestamp'])
    pipe.ltrim(f"{user_key}:timestamps", 0, 4)
    pipe.sadd(f"{user_key}:merchants", txn['merchant'])
    pipe.set(f"{user_key}:last_txn_time", txn['timestamp'])
    ttl_secs = 3600
    pipe.expire(f"{user_key}:amounts", ttl_secs)
    pipe.expire(f"{user_key}:timestamps", ttl_secs)
    pipe.expire(f"{user_key}:merchants", ttl_secs)
    pipe.expire(f"{user_key}:last_txn_time", ttl_secs)
    pipe.execute()

def extract_features(txn):
    user_key = f"user:{txn['user_id']}:features"
    try:
        amounts = list(map(float, r.lrange(f"{user_key}:amounts", 0, 4))) or []
        timestamps = r.lrange(f"{user_key}:timestamps", 0, 4)
        unique_merchants = r.scard(f"{user_key}:merchants") or 0
    except Exception as e:
        print(f"Redis read error: {e}")
        amounts, timestamps, unique_merchants = [], [], 0

    avg_txn_amt = np.mean(amounts) if amounts else txn['amount']
    var_txn_amt = np.var(amounts) if len(amounts) > 1 else 0

    now = parse_isoformat_utc(txn['timestamp'])
    txn_times = [parse_isoformat_utc(ts) for ts in timestamps if ts]
    txn_freq_1hr = sum(1 for t in txn_times if (now - t).total_seconds() <= 3600)

    try:
        last_txn_time = r.get(f"{user_key}:last_txn_time")
        time_since_last_txn = (now - parse_isoformat_utc(last_txn_time)).total_seconds() if last_txn_time else -1
    except Exception:
        time_since_last_txn = -1

    features = {
        'amount': txn['amount'],
        'time_since_last_txn': time_since_last_txn,
        'avg_txn_amt': avg_txn_amt,
        'var_txn_amt': var_txn_amt,
        'unique_merchants': unique_merchants,
        'txn_freq_1hr': txn_freq_1hr,
        
        # All expected merchants
        'merchant_Amazon': int(txn.get('merchant') == 'Amazon'),
        'merchant_AppleStore': int(txn.get('merchant') == 'AppleStore'),
        'merchant_Ebay': int(txn.get('merchant') == 'Ebay'),
        'merchant_Target': int(txn.get('merchant') == 'Target'),
        'merchant_Walmart': int(txn.get('merchant') == 'Walmart'),

        # Transaction types
        'transaction_type_purchase': int(txn.get('transaction_type') == 'purchase'),
        'transaction_type_refund': int(txn.get('transaction_type') == 'refund'),

        # Locations
        'location_DE': int(txn.get('location') == 'DE'),
        'location_FR': int(txn.get('location') == 'FR'),
        'location_IN': int(txn.get('location') == 'IN'),
        'location_UK': int(txn.get('location') == 'UK'),
        'location_US': int(txn.get('location') == 'US')
    }

    df = pd.DataFrame([features])
    return df

print("🚀 Fraud detection consumer started...")

try:
    while True:
        msg = consumer.poll(1.0)
        if msg is None:
            continue
        if msg.error():
            print(f"Kafka error: {msg.error()}")
            continue

        txn = json.loads(msg.value().decode('utf-8'))
        print(f"Received txn: {txn}")

        update_user_features(txn)
        features = extract_features(txn)
        print(features)
        risk = model.predict(features)[0]
        risk_prob = model.predict_proba(features)[0][1]

        print(f"🚨 Fraud risk score: {risk_prob:.4f} (predicted label: {risk}) | Txn ID: {txn['transaction_id']}")

except KeyboardInterrupt:
    print("🔚 Consumer shutting down...")

finally:
    consumer.close()
