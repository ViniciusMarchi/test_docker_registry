import os
import yaml
import logging
import requests
from datetime import datetime
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.exceptions import AirflowSkipException
from kubernetes import client, config

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
MODEL_NAME = os.getenv("MLFLOW_MODEL_NAME", "fraud-detection-model")

logger = logging.getLogger(__name__)


def get_latest_model_version():
    """
    Fetch the latest version of the model from the MLflow Model Registry using the REST API.
    If the model doesn't exist (RESOURCE_DOES_NOT_EXIST), skip the DAG step (AirflowSkipException).
    Otherwise, raise an error for generic 404 or other failures.
    """
    url = f"{MLFLOW_TRACKING_URI}/api/2.0/mlflow/registered-models/get-latest-versions"
    payload = {"name": MODEL_NAME}
    headers = {"Content-Type": "application/json"}

    try:
        logger.info(f"Fetching latest model version for model: {MODEL_NAME}")
        logger.info(f"MLFLOW_TRACKING_URI: {MLFLOW_TRACKING_URI}")
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()

        # Parse model versions
        model_versions = response.json().get("model_versions", [])
        if not model_versions:
            logger.warning(f"No versions found for model: {MODEL_NAME}. Skipping deployment.")
            raise AirflowSkipException(f"No versions found for model '{MODEL_NAME}'")

        # If we reach here, we have at least one model version
        latest_version = model_versions[0]["version"]
        logger.info(f"Latest model version: {latest_version}")
        return latest_version

    except requests.exceptions.HTTPError as http_err:
        if response.status_code == 404:
            # Attempt to parse MLflow error JSON
            try:
                error_json = response.json()
                error_code = error_json.get("error_code")
                if error_code == "RESOURCE_DOES_NOT_EXIST":
                    logger.warning(f"Model '{MODEL_NAME}' does not exist in MLflow. Skipping deployment.")
                    raise AirflowSkipException(f"MLflow returned RESOURCE_DOES_NOT_EXIST for model '{MODEL_NAME}'")
                else:
                    # Some other 404 reason (e.g., route not found)
                    logger.error("404 Not Found for an unexpected reason.")
                    logger.error(f"Response Body: {error_json}")
            except ValueError:
                # Could not parse JSON from the error; treat as generic 404
                logger.error("404 Not Found (route issue?). Could not parse error JSON.")
            # In either case, re-raise the HTTPError to fail the task
            raise http_err
        else:
            # Some other HTTP error (not 404 or different 404)
            logger.error(f"HTTP error: {http_err}")
            logger.error(f"Status Code: {response.status_code}")
            logger.error(f"Response Body: {response.text}")
            raise
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise


def apply_k8s_resource(resource_doc, read_func, update_func, create_func):
    """
    Generic function to apply a Kubernetes resource (e.g., Deployment or Service).
    Handles update if the resource exists or creates a new one if not.
    """
    resource_name = resource_doc["metadata"]["name"]
    namespace = resource_doc["metadata"].get("namespace", "default")
    kind = resource_doc["kind"]

    try:
        existing_resource = read_func(resource_name, namespace)
        resource_doc["metadata"]["resourceVersion"] = existing_resource.metadata.resource_version
        update_func(resource_name, namespace, resource_doc)
        logger.info(f"{kind} '{resource_name}' updated in namespace '{namespace}'.")
    except client.exceptions.ApiException as e:
        if e.status == 404:
            create_func(namespace, resource_doc)
            logger.info(f"{kind} '{resource_name}' created in namespace '{namespace}'.")
        else:
            logger.error(f"Failed to apply {kind} '{resource_name}': {e}")
            raise


def apply_deployment(**context):
    """
    Applies Kubernetes resources defined in the YAML (Deployment and Service).
    Assumes the YAML contains exactly two documents: a Deployment and a Service.
    """
    model_version = context["ti"].xcom_pull(task_ids="get_latest_model_version")
    logger.info(f"Deploying model version: {model_version}")

    # Load Kubernetes config
    config.load_incluster_config()
    core_api = client.CoreV1Api()
    apps_api = client.AppsV1Api()

    # Read ConfigMap containing the YAML definition
    cm_name = "mlflow-deployment-yaml"
    cm_ns = "airflow"
    cm = core_api.read_namespaced_config_map(name=cm_name, namespace=cm_ns)

    deployment_yaml = cm.data.get("mlflow-deployment.yaml")
    if not deployment_yaml:
        raise ValueError(f"ConfigMap '{cm_name}' does not contain 'mlflow-deployment.yaml'")

    # Replace placeholders in the YAML
    deployment_yaml = deployment_yaml.replace("{{MODEL_NAME}}", MODEL_NAME)
    deployment_yaml = deployment_yaml.replace("{{MODEL_VERSION}}", str(model_version))

    # Parse all documents from the YAML
    docs = yaml.safe_load_all(deployment_yaml)

    # Map Kubernetes kinds to the appropriate APIs
    resource_handlers = {
        "deployment": {
            "read": apps_api.read_namespaced_deployment,
            "update": apps_api.replace_namespaced_deployment,
            "create": apps_api.create_namespaced_deployment,
        },
        "service": {
            "read": core_api.read_namespaced_service,
            "update": core_api.replace_namespaced_service,
            "create": core_api.create_namespaced_service,
        },
    }

    # Loop through each document and apply it
    for doc in docs:
        if not doc:
            continue  # Skip empty documents

        kind = doc.get("kind", "").lower()
        handler = resource_handlers.get(kind)
        if handler:
            apply_k8s_resource(doc, handler["read"], handler["update"], handler["create"])
            logger.info(f"{kind.capitalize()} applied successfully.")
        else:
            logger.warning(f"Unsupported resource kind: {kind}")


default_args = {
    "depends_on_past": False,
    "start_date": datetime.now(),
}

with DAG(
    dag_id="deploy_model_dag",
    default_args=default_args,
    schedule_interval="@daily",
    catchup=False,
    description="Deploy the latest model from MLflow Model Registry (via K8s Deployment)",
) as dag:
    # Fetch the latest version from MLflow
    get_latest_version_task = PythonOperator(task_id="get_latest_model_version", python_callable=get_latest_model_version)

    # Apply or update the K8s Deployment with that version
    apply_deployment_task = PythonOperator(
        task_id="apply_deployment",
        python_callable=apply_deployment,
        provide_context=True,
    )

    get_latest_version_task >> apply_deployment_task
