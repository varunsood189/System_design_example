import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score
import joblib
import mlflow
import mlflow.xgboost
from mlflow.models.signature import infer_signature
import shap
import matplotlib.pyplot as plt


# Set experiment name and tracking URI
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("Fraud Detection Experiment")

# Load data
data = pd.read_csv('fraud_data.csv')

# One-hot encode categorical columns
categorical_cols = ['merchant', 'transaction_type', 'location']
data = pd.get_dummies(data, columns=categorical_cols, drop_first=False)

# Prepare features and labels
X = data.drop(['is_fraud', 'transaction_id', 'user_id', 'timestamp'], axis=1)
y = data['is_fraud']

# Infer signature for model schema
signature = infer_signature(X, y)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(X_train.columns)
# Enable autologging
mlflow.xgboost.autolog()
reference = pd.read_csv("fraud_data.csv").drop(
    ["is_fraud", "transaction_id", "user_id", "timestamp"], axis=1
)
reference = pd.get_dummies(reference, columns=categorical_cols, drop_first=False)
reference = reference.astype(int)  # Convert boolean to 0/1

reference.to_csv("reference.csv", index=False)

# Start MLflow run
with mlflow.start_run(run_name="xgb-fraud-model") as run:
    model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    model.fit(X_train, y_train)

    # Predict & evaluate
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)

    # Log additional metrics
    mlflow.log_metric("precision", precision_score(y_test, y_pred))
    mlflow.log_metric("recall", recall_score(y_test, y_pred))
    mlflow.log_metric("f1_score", f1_score(y_test, y_pred))

    # Print classification report to console
    print(classification_report(y_test, y_pred))

    # Log the model explicitly with signature
    mlflow.xgboost.log_model(
        model,
        artifact_path="fraud_xgb_model",
        signature=signature,
        registered_model_name="xgb-fraud-model"
    )


    # Optionally promote to production manually
    print("Model training complete and registered in MLflow.")

    # Save locally too
    joblib.dump(model, 'fraud_model.pkl')
    

    # Initialize SHAP explainer
    explainer = shap.Explainer(model)
    shap_values = explainer(X_test)

    # Plot and save summary
    plt.figure()
    shap.summary_plot(shap_values, X_test, show=False)
    plt.tight_layout()
    plt.savefig("shap_summary_plot.png")

    # Log plot to MLflow
    mlflow.log_artifact("shap_summary_plot.png")

