import os
from datetime import datetime
from airflow import DAG
from airflow.providers.cncf.kubernetes.operators.pod import KubernetesPodOperator
from kubernetes.client import models as k8s

MINIO_ENDPOINT = os.environ.get("MINIO_ENDPOINT")
MINIO_ACCESS_KEY = os.environ.get("MINIO_ACCESS_KEY")
MINIO_SECRET_KEY = os.environ.get("MINIO_SECRET_KEY")
MINIO_BUCKET_RAW = os.environ.get("MINIO_BUCKET_RAW")
MINIO_BUCKET_PROCESSED = os.environ.get("MINIO_BUCKET_PROCESSED")
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI")
MLFLOW_EXPERIMENT_NAME = os.environ.get("MLFLOW_EXPERIMENT_NAME")
MLFLOW_MODEL_NAME = os.environ.get("MLFL0W_MODEL_NAME", "fraud-detection-model")

default_args = {
    "depends_on_past": False,
    "start_date": datetime.now(),
}

with DAG(
    dag_id="etl_train_dag",
    default_args=default_args,
    schedule_interval="@daily",
    catchup=False,
    description="A DAG to ingest and store raw data in MinIO",
) as dag:
    ingest_task = KubernetesPodOperator(
        task_id="ingest_data",
        namespace="airflow",
        name="data-ingestion-pod",
        image="data_ingestion:latest",
        cmds=["python", "data_ingestion.py"],
        env_vars={
            "MINIO_ENDPOINT": MINIO_ENDPOINT,
            "MINIO_ACCESS_KEY": MINIO_ACCESS_KEY,
            "MINIO_SECRET_KEY": MINIO_SECRET_KEY,
            "MINIO_BUCKET_RAW": MINIO_BUCKET_RAW,
        },
        get_logs=True,
        is_delete_operator_pod=True,
        image_pull_policy="IfNotPresent",
    )

    data_preparation_task = KubernetesPodOperator(
        task_id="data_preparation",
        namespace="airflow",
        name="data-preparation-pod",
        image="data_preparation:latest",
        cmds=["python", "data_preparation.py"],
        env_vars={
            "MINIO_ENDPOINT": MINIO_ENDPOINT,
            "MINIO_ACCESS_KEY": MINIO_ACCESS_KEY,
            "MINIO_SECRET_KEY": MINIO_SECRET_KEY,
            "MINIO_BUCKET_RAW": MINIO_BUCKET_RAW,
            "MINIO_BUCKET_PROCESSED": MINIO_BUCKET_PROCESSED,
        },
        get_logs=True,
        is_delete_operator_pod=True,
        image_pull_policy="IfNotPresent",
    )

    train_task = KubernetesPodOperator(
        task_id="train_models",
        namespace="airflow",
        name="train-models-pod",
        image="train:latest",
        cmds=["python", "train.py"],
        env_vars={
            "MINIO_ENDPOINT": MINIO_ENDPOINT,
            "MINIO_ACCESS_KEY": MINIO_ACCESS_KEY,
            "MINIO_SECRET_KEY": MINIO_SECRET_KEY,
            "MINIO_BUCKET_RAW": MINIO_BUCKET_RAW,
            "MINIO_BUCKET_PROCESSED": MINIO_BUCKET_PROCESSED,
            "MLFLOW_TRACKING_URI": MLFLOW_TRACKING_URI,
            "MLFLOW_EXPERIMENT_NAME": MLFLOW_EXPERIMENT_NAME,
            "MLFLOW_MODEL_NAME": MLFLOW_MODEL_NAME,
        },
        get_logs=True,
        is_delete_operator_pod=True,
        image_pull_policy="IfNotPresent",
        container_resources=k8s.V1ResourceRequirements(
            requests={"memory": "2Gi", "cpu": "1"},
            limits={"memory": "4Gi", "cpu": "2"},
        ),
    )

    ingest_task >> data_preparation_task >> train_task
