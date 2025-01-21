import streamlit as st
import pandas as pd
from utils import (
    create_minio_client,
    load_scaler_from_minio,
    transform_input,
    predict,
    MINIO_BUCKET_PROCESSED,
    CATEGORIES,
)


def main():
    st.header("Single Transaction Inference")

    # Create MinIO client & load scaler
    client = create_minio_client()
    scaler = load_scaler_from_minio(client, MINIO_BUCKET_PROCESSED, "scaler.pkl")

    # Create two columns for input
    col1, col2 = st.columns(2)
    with col1:
        amt = st.number_input("Transaction Amount", min_value=0.0, value=75.0)
        gender = st.selectbox("Gender", ["M", "F"])
        zip_code = st.number_input("ZIP Code", min_value=0, value=12345)
        lat = st.number_input("Latitude", value=39.95)

    with col2:
        longi = st.number_input("Longitude", value=-75.16)
        city_pop = st.number_input("City Population", min_value=0, value=50000)
        merch_lat = st.number_input("Merchant Lat", value=39.96)
        merch_long = st.number_input("Merchant Long", value=-75.14)

    category = st.selectbox("Category", CATEGORIES)

    if st.button("Predict Fraud"):
        raw_dict = {
            "amt": amt,
            "gender": gender,
            "zip": zip_code,
            "lat": lat,
            "long": longi,
            "city_pop": city_pop,
            "merch_lat": merch_lat,
            "merch_long": merch_long,
            "category": category,
        }
        df_input = pd.DataFrame([raw_dict])
        df_transformed = transform_input(df_input, scaler)

        try:
            preds = predict(df_transformed)
            pred_class = preds[0]
            if pred_class == 1:
                st.warning("Fraud predicted! (High recall => fewer false negatives.)")
            else:
                st.success("Not Fraud.")
        except Exception as e:
            st.error(f"Error calling model: {e}")


if __name__ == "__main__":
    main()
