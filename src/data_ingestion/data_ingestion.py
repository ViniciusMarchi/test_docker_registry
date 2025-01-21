import os
import logging
import requests
import tempfile
import zipfile
from pathlib import Path
from minio import Minio
from minio.error import S3Error

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

KAGGLE_DATA_URL = (
    "https://www.kaggle.com/api/v1/datasets/download/kartik2112/fraud-detection"
)

MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY")
MINIO_BUCKET_RAW = os.getenv("MINIO_BUCKET_RAW")


def download_dataset(download_url: str, local_filepath: str) -> None:
    logger.info(f"Starting download from: {download_url}")
    response = requests.get(download_url, stream=True)
    response.raise_for_status()

    with open(local_filepath, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
    logger.info(f"Dataset downloaded to {local_filepath}")


# Extracts a zip file to a specified directory.
def extract_zip(zip_filepath: str, extract_to: str) -> None:
    logger.info(f"Extracting {zip_filepath} to {extract_to}")
    with zipfile.ZipFile(zip_filepath, "r") as zip_ref:
        zip_ref.extractall(extract_to)
    logger.info(f"Extraction complete. Files extracted to {extract_to}")


# Recursively uploads all files in a directory to a specified MinIO bucket.
def upload_directory_to_minio(
    directory: str, bucket_name: str, minio_client: Minio
) -> None:
    for root, _, files in os.walk(directory):
        for file in files:
            local_file_path = Path(root) / file
            object_name = str(local_file_path.relative_to(directory))
            logger.info(f"Uploading {local_file_path} as {object_name} to MinIO...")
            try:
                minio_client.fput_object(bucket_name, object_name, str(local_file_path))
            except S3Error as exc:
                logger.error(f"Error uploading {object_name} to MinIO: {exc}")
                raise


def main():
    minio_client = Minio(
        MINIO_ENDPOINT,
        access_key=MINIO_ACCESS_KEY,
        secret_key=MINIO_SECRET_KEY,
        secure=False,
    )

    # Ensure the bucket exists
    if not minio_client.bucket_exists(MINIO_BUCKET_RAW):
        logger.info(f"Bucket '{MINIO_BUCKET_RAW}' not found. Creating it.")
        minio_client.make_bucket(MINIO_BUCKET_RAW)

    # Download the dataset to a temporary file and load into minio
    with tempfile.TemporaryDirectory() as temp_dir:
        zip_filepath = Path(temp_dir) / "dataset.zip"
        extract_dir = Path(temp_dir) / "extracted"
        extract_dir.mkdir()

        download_dataset(KAGGLE_DATA_URL, zip_filepath)
        extract_zip(zip_filepath, str(extract_dir))
        upload_directory_to_minio(str(extract_dir), MINIO_BUCKET_RAW, minio_client)


if __name__ == "__main__":
    main()
