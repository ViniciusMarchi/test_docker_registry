import sys
import logging
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from minio_utils import (
    create_minio_client,
    read_csv_from_minio,
    write_csv_to_minio,
    write_pickle_to_minio,
    MINIO_BUCKET_RAW,
    MINIO_BUCKET_PROCESSED,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def clean_raw_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Minimal domain-level cleaning:
      - Drop duplicates, etc.
    """
    logger.info(f"Initial shape: {df.shape}")
    df.drop_duplicates(inplace=True)
    logger.info(f"Shape after dropping duplicates: {df.shape}")
    return df


def gender_transform(g):
    """
    Convert gender 'F' -> 1, 'M' -> 0; else None/NaN.
    """
    if g == "F":
        return 1
    elif g == "M":
        return 0
    return None


def apply_feature_transforms(X: pd.DataFrame, numeric_scaler: StandardScaler = None, fit_scaler: bool = False) -> pd.DataFrame:
    """
    Apply transformations to a feature DataFrame X:
      - gender -> numeric
      - one-hot encode 'category'
      - drop unused columns
      - scale numeric columns (fit_scaler=True only for train)
    Returns the transformed X and optionally modifies the scaler in-place if fitting.
    """
    df = X.copy()

    # Gender
    if "gender" in df.columns:
        df["gender"] = df["gender"].apply(gender_transform)

    # One-hot for 'category'
    if "category" in df.columns:
        cat_dummies = pd.get_dummies(df["category"], prefix="category", dtype=int)
        df.drop("category", axis=1, inplace=True)
        df = pd.concat([df, cat_dummies], axis=1)

    # Drop columns not used
    drop_cols = [
        "Unnamed: 0",
        "unix_time",
        "trans_date_trans_time",
        "cc_num",
        "trans_num",
        "street",
        "dob",
        "city",
        "merchant",
        "job",
        "last",
        "first",
        "state",
    ]
    for c in drop_cols:
        if c in df.columns:
            df.drop(c, axis=1, inplace=True)

    # Scale numeric columns
    numeric_cols = ["amt", "lat", "long", "city_pop", "merch_lat", "merch_long", "zip"]
    numeric_cols = [col for col in numeric_cols if col in df.columns]
    if numeric_scaler is not None and len(numeric_cols) > 0:
        if fit_scaler:
            numeric_scaler.fit(df[numeric_cols])
        df[numeric_cols] = numeric_scaler.transform(df[numeric_cols])

    return df


def split_and_transform(df: pd.DataFrame):
    """
    Splits the cleaned raw DataFrame into train/val/test, then applies
    transformations, scaling (StandardScaler), and SMOTE on train only.

    Returns:
      X_train_fe, y_train_fe,
      X_val_fe, y_val_fe,
      X_test_fe, y_test_fe,
      fitted_scaler
    """

    # split dataset
    y = df["is_fraud"].values
    X = df.drop("is_fraud", axis=1)
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, stratify=y, test_size=0.4, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, stratify=y_temp, test_size=0.5, random_state=42)

    logger.info(f"Train shape: {X_train.shape}, Val shape: {X_val.shape}, Test shape: {X_test.shape}")

    # scale train numeric features
    numeric_scaler = StandardScaler()
    X_train_fe = apply_feature_transforms(X_train, numeric_scaler, fit_scaler=True)
    # scale val/test numeric features using the same fitted scaler
    X_val_fe = apply_feature_transforms(X_val, numeric_scaler, fit_scaler=False)
    X_test_fe = apply_feature_transforms(X_test, numeric_scaler, fit_scaler=False)

    # perform SMOTE on train due imbalanced dataset
    sm = SMOTE(random_state=42)
    X_train_fe, y_train_fe = sm.fit_resample(X_train_fe, y_train)

    return X_train_fe, y_train_fe, X_val_fe, y_val, X_test_fe, y_test, numeric_scaler


def main():
    client = create_minio_client()
    df_raw = read_csv_from_minio(client, MINIO_BUCKET_RAW, "fraudTrain.csv")
    df_cleaned = clean_raw_data(df_raw)

    (
        X_train_fe,
        y_train_fe,
        X_val_fe,
        y_val_fe,
        X_test_fe,
        y_test_fe,
        fitted_scaler,
    ) = split_and_transform(df_cleaned)

    write_csv_to_minio(client, X_train_fe, MINIO_BUCKET_PROCESSED, "X_train_fe.csv")
    write_csv_to_minio(
        client,
        pd.DataFrame({"is_fraud": y_train_fe}),
        MINIO_BUCKET_PROCESSED,
        "y_train.csv",
    )

    write_csv_to_minio(client, X_val_fe, MINIO_BUCKET_PROCESSED, "X_val_fe.csv")
    write_csv_to_minio(
        client,
        pd.DataFrame({"is_fraud": y_val_fe}),
        MINIO_BUCKET_PROCESSED,
        "y_val.csv",
    )

    write_csv_to_minio(client, X_test_fe, MINIO_BUCKET_PROCESSED, "X_test_fe.csv")
    write_csv_to_minio(
        client,
        pd.DataFrame({"is_fraud": y_test_fe}),
        MINIO_BUCKET_PROCESSED,
        "y_test.csv",
    )

    write_pickle_to_minio(client, fitted_scaler, MINIO_BUCKET_PROCESSED, "scaler.pkl")

    logger.info("Data preparation complete")


if __name__ == "__main__":
    main()
