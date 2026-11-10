
import streamlit as st
import pandas as pd

from core.prediction import (
    predict_new_data,
    download_predictions
)


st.set_page_config(
    page_title="Predictions | ML Studio",
    page_icon="🔮",
    layout="wide"
)

st.title("🔮 Model Predictions")
st.write(
    "Generate predictions for new datasets using your trained model."
)

# ---------------- CHECK TRAINED MODEL ----------------

if "training_result" not in st.session_state:
    st.warning(
        "Please train a model from the Model Training page first."
    )
    st.stop()

result = st.session_state["training_result"]

best_model = result["best_model"]
target_column = result["target"]

st.success(
    f"Active model: {result['best_model_name']}"
)

st.caption(f"Prediction target: {target_column}")

st.divider()

# ---------------- UPLOAD NEW DATA ----------------

st.header("Upload New Dataset")

uploaded_file = st.file_uploader(
    "Upload CSV or Excel data for prediction",
    type=["csv", "xlsx", "xls"]
)

if uploaded_file is not None:

    try:
        file_name = uploaded_file.name.lower()

        if file_name.endswith(".csv"):
            new_data = pd.read_csv(uploaded_file)

        elif file_name.endswith((".xlsx", ".xls")):
            new_data = pd.read_excel(uploaded_file)

        else:
            raise ValueError("Unsupported file format.")

        if new_data.empty:
            raise ValueError("The uploaded dataset is empty.")

        st.session_state["prediction_data"] = new_data

        st.success("New dataset uploaded successfully!")

    except Exception as error:
        st.error(f"Upload failed: {error}")


# ---------------- DATA PREVIEW ----------------

if "prediction_data" in st.session_state:

    new_data = st.session_state["prediction_data"]

    st.subheader("New Dataset Preview")

    st.dataframe(
        new_data.head(10),
        use_container_width=True
    )

    st.write(
        f"Rows: {len(new_data)} | Columns: {len(new_data.columns)}"
    )

    # ---------------- GENERATE PREDICTIONS ----------------

    st.divider()

    if st.button(
        "✨ Generate Predictions",
        type="primary",
        use_container_width=True
    ):

        try:
            training_data = st.session_state["dataset"]

            expected_columns = [
                column
                for column in training_data.columns
                if column != target_column
            ]

            with st.spinner("Generating predictions..."):

                prediction_result = predict_new_data(
                    model=best_model,
                    new_data=new_data,
                    expected_columns=expected_columns,
                    target_column=target_column
                )

                st.session_state["prediction_result"] = (
                    prediction_result
                )

            st.success("Predictions generated successfully!")

        except Exception as error:
            st.error(f"Prediction failed: {error}")


# ---------------- DISPLAY RESULTS ----------------

if "prediction_result" in st.session_state:

    prediction_result = st.session_state["prediction_result"]

    st.divider()

    st.header("Prediction Results")

    st.dataframe(
        prediction_result,
        use_container_width=True
    )

    st.metric(
        "Total Predictions Generated",
        len(prediction_result)
    )

    # ---------------- DOWNLOAD ----------------

    csv_data = download_predictions(prediction_result)

    st.download_button(
        label="⬇️ Download Predictions as CSV",
        data=csv_data,
        file_name="ml_studio_predictions.csv",
        mime="text/csv",
        use_container_width=True
    )


st.divider()

st.caption("ML Studio | Prediction and Export Module")
