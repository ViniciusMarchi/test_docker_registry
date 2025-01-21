import os
import logging
import pandas as pd
from io import BytesIO
from minio import Minio
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, recall_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from mlflow.models.signature import infer_signature

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY")
MINIO_BUCKET_PROCESSED = os.getenv("MINIO_BUCKET_PROCESSED")

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
MLFLOW_EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "fraud-detection-experiment")
MODEL_NAME = os.getenv("MLFLOW_MODEL_NAME", "fraud-detection-model")


def fetch_csv(client, bucket, obj_name):
    resp = client.get_object(bucket, obj_name)
    return pd.read_csv(BytesIO(resp.read()))


def main():
    client = Minio(
        MINIO_ENDPOINT,
        access_key=MINIO_ACCESS_KEY,
        secret_key=MINIO_SECRET_KEY,
        secure=False,
    )

    # Load feature-engineered data
    logger.info("Fetching preprocessed train/val from MinIO...")
    X_train = fetch_csv(client, MINIO_BUCKET_PROCESSED, "X_train_fe.csv")
    y_train = fetch_csv(client, MINIO_BUCKET_PROCESSED, "y_train.csv")["is_fraud"]

    X_val = fetch_csv(client, MINIO_BUCKET_PROCESSED, "X_val_fe.csv")
    y_val = fetch_csv(client, MINIO_BUCKET_PROCESSED, "y_val.csv")["is_fraud"]

    # MLflow setup
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    # Model candidates Optimized for recall
    model_candidates = [
        (
            "LogisticRegression",
            LogisticRegression(solver="liblinear"),
            {
                "C": [0.1, 1.0, 10.0],
                "max_iter": [200, 400],
            },
        ),
        (
            "RandomForest",
            RandomForestClassifier(random_state=42),
            {
                "n_estimators": [50, 100],
                "max_depth": [None, 10],
            },
        ),
    ]

    best_run_id = None
    best_recall = 0.0

    for model_name, model_obj, param_grid in model_candidates:
        with mlflow.start_run(run_name=f"{model_name}-GridSearch"):
            mlflow.log_param("model_name", model_name)

            grid_search = GridSearchCV(
                estimator=model_obj,
                param_grid=param_grid,
                scoring="recall",
                cv=3,
                n_jobs=2,
                verbose=1,
            )
            logger.info(f"Running GridSearch with recall for {model_name} ...")
            grid_search.fit(X_train, y_train)

            best_model = grid_search.best_estimator_
            best_params = grid_search.best_params_
            best_cv_recall = grid_search.best_score_

            logger.info(f"[{model_name}] Best Params: {best_params}")
            logger.info(f"[{model_name}] CV Recall: {best_cv_recall:.3f}")

            # Evaluate on validation
            val_preds = best_model.predict(X_val)
            val_accuracy = accuracy_score(y_val, val_preds)
            val_f1 = f1_score(y_val, val_preds)
            val_recall = recall_score(y_val, val_preds)

            mlflow.log_metric("val_accuracy", val_accuracy)
            mlflow.log_metric("val_f1_score", val_f1)
            mlflow.log_metric("val_recall", val_recall)

            cm = confusion_matrix(y_val, val_preds)
            logger.info(f"[{model_name}] Confusion Matrix:\n{cm}")
            logger.info(f"[{model_name}] Val Accuracy={val_accuracy:.3f}, Val F1={val_f1:.3f}, Val Recall={val_recall:.3f}")

            signature = infer_signature(X_train, best_model.predict(X_train))
            mlflow.sklearn.log_model(best_model, "model", signature=signature)

            # Track best recall
            if val_recall > best_recall:
                best_recall = val_recall
                best_run_id = mlflow.active_run().info.run_id

    # Register best run
    if best_run_id:
        logger.info(f"Best run found by recall: run_id={best_run_id}, recall={best_recall:.3f}")
        ml_client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
        ml_client.set_tag(best_run_id, "best_run", "True")

        model_uri = f"runs:/{best_run_id}/model"
        model_details = mlflow.register_model(model_uri, MODEL_NAME)
        logger.info(f"Registered model '{model_details.name}' version {model_details.version}")
    else:
        logger.warning("No best run found.")


if __name__ == "__main__":
    main()
