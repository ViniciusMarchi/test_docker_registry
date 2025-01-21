import streamlit as st
import pandas as pd
from io import BytesIO
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, f1_score

from utils import (
    create_minio_client,
    load_scaler_from_minio,
    transform_input,
    predict,
    MINIO_BUCKET_RAW,
    MINIO_BUCKET_PROCESSED,
)


def main():
    st.set_page_config(page_title="Batch Inference", page_icon=":bar_chart:")
    st.header("Batch Inference on Multiple Test Cases")

    st.markdown(
        "Click the **Generate Multiple Cases** button to automatically download "
        "`fraudTest.csv` from the **raw** bucket, run batch inferences, and display results."
    )

    client = create_minio_client()
    scaler = load_scaler_from_minio(client, MINIO_BUCKET_PROCESSED, "scaler.pkl")

    if st.button("Generate Multiple Cases"):
        with st.spinner("Loading test data from MinIO and predicting..."):
            try:
                obj = client.get_object(MINIO_BUCKET_RAW, "fraudTest.csv")
                df_test_raw = pd.read_csv(BytesIO(obj.read()))
            except Exception as e:
                st.error(f"Error reading `fraudTest.csv` from MinIO (raw bucket): {e}")
                return

            if "is_fraud" not in df_test_raw.columns:
                st.error("The test file does not contain 'is_fraud' column.")
                return

            y_test = df_test_raw["is_fraud"].values
            X_test = df_test_raw.drop("is_fraud", axis=1)

            X_test_transformed = transform_input(X_test.copy(), scaler)

            try:
                preds = predict(X_test_transformed)
            except Exception as e:
                st.error(f"Error calling model: {e}")
                return

        st.success("Batch inference complete!")

        df_result = X_test.copy()
        df_result["actual"] = y_test
        df_result["predicted"] = preds

        st.subheader("Evaluation Metrics")
        accuracy = accuracy_score(y_test, preds)
        recall = recall_score(y_test, preds)
        f1 = f1_score(y_test, preds)

        st.write(f"**Accuracy**: {accuracy:.4f}")
        st.write(f"**Recall**: {recall:.4f}")
        st.write(f"**F1-score**: {f1:.4f}")

        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y_test, preds)
        fig_cm, ax_cm = plt.subplots()
        sns.heatmap(cm, annot=True, cmap="Blues", fmt="d", ax=ax_cm)
        ax_cm.set_xlabel("Predicted")
        ax_cm.set_ylabel("Actual")
        ax_cm.set_title("Confusion Matrix")
        st.pyplot(fig_cm)

        st.subheader("Distribution of Predicted Fraud vs. Non-Fraud")
        df_result["predicted_label"] = np.where(df_result["predicted"] == 1, "Fraud", "Not Fraud")
        pred_counts = df_result["predicted_label"].value_counts()

        fig_pie, ax_pie = plt.subplots(figsize=(5, 5))
        ax_pie.pie(pred_counts, labels=pred_counts.index, autopct="%1.1f%%", startangle=140)
        ax_pie.set_title("Predicted Fraud Distribution")
        st.pyplot(fig_pie)

        if "category" in df_result.columns and "amt" in df_result.columns:
            st.subheader("Transaction Amount by Category and Predicted Fraud Label")

            fig_box, ax_box = plt.subplots(figsize=(10, 5))
            sns.boxplot(
                x="category",
                y="amt",
                hue="predicted_label",
                data=df_result,
                palette="Set2",
                ax=ax_box,
            )
            ax_box.set_title("Amount by Category (split by Predicted Label)")
            plt.setp(ax_box.get_xticklabels(), rotation=45, ha="right")
            st.pyplot(fig_box)
        else:
            st.info("No `category` or `amt` column available for box plot. Check if your dataset includes those columns.")

        st.subheader("Count of Predicted Fraud by Category")
        if "category" in df_result.columns:
            cat_fraud_counts = df_result[df_result["predicted"] == 1]["category"].value_counts().sort_values(ascending=False)
            # Create horizontal bar plot
            fig_bar, ax_bar = plt.subplots(figsize=(7, 5))
            sns.barplot(
                y=cat_fraud_counts.index,  # categories on y-axis
                x=cat_fraud_counts.values,  # counts on x-axis
                color="salmon",
                ax=ax_bar,
            )
            ax_bar.set_title("Predicted Fraud Count by Category")
            ax_bar.set_xlabel("Count of Fraud Predictions")
            ax_bar.set_ylabel("Category")

            st.pyplot(fig_bar)
        else:
            st.info("No `category` column to visualize fraud counts by category.")


if __name__ == "__main__":
    main()
