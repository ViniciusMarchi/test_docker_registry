import os
import requests
import pickle
import pandas as pd
import numpy as np
from minio import Minio

MLFLOW_SERVE_URL = os.getenv("MLFLOW_SERVE_DEPLOYMENT_URI")
MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY")
MINIO_BUCKET_RAW = os.getenv("MINIO_BUCKET_RAW")
MINIO_BUCKET_PROCESSED = os.environ.get("MINIO_BUCKET_PROCESSED")

CATEGORIES = [
    "entertainment",
    "food_dining",
    "gas_transport",
    "grocery_net",
    "grocery_pos",
    "health_fitness",
    "home",
    "kids_pets",
    "misc_net",
    "misc_pos",
    "personal_care",
    "shopping_net",
    "shopping_pos",
    "travel",
]

MLFLOW_SCHEMA = {
    "amt": "float64",
    "gender": "int64",
    "zip": "float64",
    "lat": "float64",
    "long": "float64",
    "city_pop": "float64",
    "merch_lat": "float64",
    "merch_long": "float64",
    "category_entertainment": "int64",
    "category_food_dining": "int64",
    "category_gas_transport": "int64",
    "category_grocery_net": "int64",
    "category_grocery_pos": "int64",
    "category_health_fitness": "int64",
    "category_home": "int64",
    "category_kids_pets": "int64",
    "category_misc_net": "int64",
    "category_misc_pos": "int64",
    "category_personal_care": "int64",
    "category_shopping_net": "int64",
    "category_shopping_pos": "int64",
    "category_travel": "int64",
}


def create_minio_client():
    return Minio(
        MINIO_ENDPOINT,
        access_key=MINIO_ACCESS_KEY,
        secret_key=MINIO_SECRET_KEY,
        secure=False,
    )


def load_scaler_from_minio(client: Minio, bucket: str, scaler_file: str):
    obj = client.get_object(bucket, scaler_file)
    return pickle.loads(obj.read())


def apply_schema_to_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    for col, dtype in MLFLOW_SCHEMA.items():
        if col not in df.columns:
            df[col] = 0
        if "int" in dtype:
            df[col] = df[col].fillna(0).astype(dtype)
        else:
            df[col] = df[col].astype(dtype)
    return df


def transform_input(df: pd.DataFrame, scaler) -> pd.DataFrame:
    df = df.copy()
    if "gender" in df.columns:
        df["gender"] = df["gender"].apply(lambda g: 1 if g == "F" else 0)
    if "category" in df.columns:
        for cat in CATEGORIES:
            df[f"category_{cat}"] = (df["category"] == cat).astype(int)
        df.drop("category", axis=1, inplace=True)
    numeric_cols = ["amt", "lat", "long", "city_pop", "merch_lat", "merch_long", "zip"]
    existing = [c for c in numeric_cols if c in df.columns]
    df[existing] = scaler.transform(df[existing])
    return apply_schema_to_dataframe(df)


def predict(df: pd.DataFrame):
    payload = {"dataframe_split": {"columns": list(df.columns), "data": df.values.tolist()}}
    response = requests.post(MLFLOW_SERVE_URL, json=payload)
    response.raise_for_status()
    return np.array(response.json()["predictions"])
