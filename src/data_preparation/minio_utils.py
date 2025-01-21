import os
import sys
import pickle
import logging
import pandas as pd
from io import BytesIO
from minio import Minio

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY")
MINIO_BUCKET_RAW = os.getenv("MINIO_BUCKET_RAW")
MINIO_BUCKET_PROCESSED = os.getenv("MINIO_BUCKET_PROCESSED")


def create_minio_client():
    """
    Creates and returns a Minio client using environment variables.
    """
    return Minio(
        MINIO_ENDPOINT,
        access_key=MINIO_ACCESS_KEY,
        secret_key=MINIO_SECRET_KEY,
        secure=False,
    )


def read_csv_from_minio(client: Minio, bucket: str, file_name: str) -> pd.DataFrame:
    """
    Reads a CSV object from MinIO and returns it as a pandas DataFrame.
    """
    logger.info(f"Reading '{file_name}' from bucket '{bucket}'...")
    obj = client.get_object(bucket, file_name)
    df = pd.read_csv(BytesIO(obj.read()))
    return df


def write_csv_to_minio(client: Minio, df: pd.DataFrame, bucket: str, file_name: str):
    """
    Writes a pandas DataFrame to a CSV file in MinIO.
    """
    logger.info(f"Writing '{file_name}' to bucket '{bucket}'...")
    if not client.bucket_exists(bucket):
        logger.info(f"Bucket '{bucket}' not found. Creating it.")
        client.make_bucket(bucket)

    csv_data = df.to_csv(index=False).encode("utf-8")
    client.put_object(
        bucket,
        file_name,
        data=BytesIO(csv_data),
        length=len(csv_data),
        content_type="text/csv",
    )


def write_pickle_to_minio(client: Minio, obj_to_pickle, bucket: str, file_name: str):
    """
    Serializes a Python object (e.g., fitted scaler) to pickle and writes it to MinIO.
    """
    logger.info(f"Writing pickle '{file_name}' to bucket '{bucket}'...")
    pkl_bytes = BytesIO()
    pickle.dump(obj_to_pickle, pkl_bytes)
    pkl_bytes.seek(0)
    client.put_object(
        bucket,
        file_name,
        data=pkl_bytes,
        length=len(pkl_bytes.getvalue()),
        content_type="application/octet-stream",
    )
