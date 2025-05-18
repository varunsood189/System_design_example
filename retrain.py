import os
import glob
import pandas as pd
import xgboost as xgb
import joblib
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score
import mlflow
import mlflow.xgboost
from mlflow.models.signature import infer_signature

# Set MLflow experiment
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("Fraud Detection Experiment")

# Load base (labeled) data
base_data = pd.read_csv("fraud_data.csv")

# One-hot encode categorical columns
categorical_cols = ['merchant', 'transaction_type', 'location']
base_data = pd.get_dummies(base_data, columns=categorical_cols, drop_first=False)

# Drop rows with missing labels
base_data = base_data.dropna(subset=["is_fraud"])

# Extract features and labels
X_base = base_data.drop(['is_fraud', 'transaction_id', 'user_id', 'timestamp'], axis=1)
X_base = X_base.apply(pd.to_numeric, errors='coerce')
y_base = base_data['is_fraud']

# Load new unlabeled data
new_files = glob.glob("logged_batches/*.csv")
if not new_files:
    print("⚠️ No new data to retrain on.")
    exit()

new_data = pd.concat([pd.read_csv(f) for f in new_files])

# Align columns: fill missing columns from base with 0, drop extras
X_new = new_data.reindex(columns=X_base.columns, fill_value=0)

# Combine only the features for training
X_final = pd.concat([X_base, X_new], ignore_index=True)
y_final = pd.concat([y_base, pd.Series([0] * len(X_new))], ignore_index=True)  # default new data as non-fraud

# Filter to use only base data (labeled) for supervised training
X_train_final = X_final.iloc[:len(X_base)]
y_train_final = y_final.iloc[:len(X_base)]

# Infer model input/output signature
signature = infer_signature(X_train_final, y_train_final)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_train_final, y_train_final, test_size=0.2, random_state=42)

# Train model
with mlflow.start_run(run_name="xgb-fraud-retrain") as run:
    model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Log metrics
    mlflow.log_metric("precision", precision_score(y_test, y_pred))
    mlflow.log_metric("recall", recall_score(y_test, y_pred))
    mlflow.log_metric("f1_score", f1_score(y_test, y_pred))

    print("🔁 Retrained model metrics:")
    print(classification_report(y_test, y_pred))

    # Save model
    mlflow.xgboost.log_model(model, "model", signature=signature, registered_model_name="xgb-fraud-model")
    joblib.dump(model, "fraud_model.pkl")

    # Archive processed files
    for file in new_files:
        archived_path = file.replace("logged_batches/", "archived_data/")
        os.makedirs(os.path.dirname(archived_path), exist_ok=True)
        os.rename(file, archived_path)

from mlflow.tracking import MlflowClient

client = MlflowClient()
latest_model = client.get_latest_versions("xgb-fraud-model", stages=["None"])[0]

# Move previous Production to Archived (optional)
try:
    current_prod = client.get_latest_versions("xgb-fraud-model", stages=["Production"])[0]
    client.transition_model_version_stage(
        name="xgb-fraud-model",
        version=current_prod.version,
        stage="Archived"
    )
except:
    pass  # No Production model yet

# Promote new model to Production
client.transition_model_version_stage(
    name="xgb-fraud-model",
    version=latest_model.version,
    stage="Production"
)

print(f"✅ Promoted version {latest_model.version} to Production.")
