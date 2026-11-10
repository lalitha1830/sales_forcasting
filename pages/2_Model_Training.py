
import streamlit as st

from core.auto_ml import train_automl
from core.model_registry import get_available_models
from utils.visualizations import plot_model_comparison


st.set_page_config(
    page_title="Model Training | ML Studio",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Model Training")
st.write(
    "Train machine learning models, compare their performance "
    "and identify the best-performing model."
)

# ---------------- DATASET CHECK ----------------

if "dataset" not in st.session_state:
    st.warning("Please upload a dataset from the homepage first.")
    st.stop()

df = st.session_state["dataset"]

st.success(
    f"Dataset loaded: {st.session_state.get('dataset_name', 'Dataset')}"
)

st.divider()

# ---------------- TRAINING CONFIGURATION ----------------

st.header("Configure Your Model")

task = st.selectbox(
    "Select Machine Learning Task",
    ["Regression", "Classification"]
)

target_column = st.selectbox(
    "Select Target Column",
    options=df.columns.tolist()
)

available_models = get_available_models(task)

if not available_models:
    st.error("No compatible models are currently available.")
    st.stop()

selection_mode = st.radio(
    "How would you like to select models?",
    [
        "AI choose for me (compare all available models)",
        "Choose models manually"
    ]
)

if selection_mode == "Choose models manually":

    selected_models = st.multiselect(
        "Select algorithms",
        options=available_models,
        default=available_models[:min(3, len(available_models))]
    )

else:
    selected_models = None

test_size = st.slider(
    "Test data percentage",
    min_value=10,
    max_value=40,
    value=20,
    step=5
)

st.caption(
    "The remaining data is used for training. "
    "The test data is used to evaluate model performance."
)

# ---------------- MODEL TRAINING ----------------

st.divider()

if st.button(
    "🚀 Start Model Training",
    type="primary",
    use_container_width=True
):

    if target_column is None:
        st.error("Please select a target column.")

    elif selected_models == []:
        st.error("Please select at least one algorithm.")

    else:

        try:
            with st.spinner(
                "Training models... Please wait."
            ):

                result = train_automl(
                    df=df,
                    target_column=target_column,
                    task=task,
                    selected_models=selected_models,
                    test_size=test_size / 100
                )

                st.session_state["training_result"] = result
                st.session_state["training_task"] = task

            st.success("Model training completed!")

        except Exception as error:
            st.error(f"Training failed: {error}")


# ---------------- TRAINING RESULTS ----------------

if "training_result" in st.session_state:

    result = st.session_state["training_result"]

    st.divider()

    st.header("🏆 Training Results")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Best Model",
        result["best_model_name"]
    )

    col2.metric(
        "Training Records",
        result["train_rows"]
    )

    col3.metric(
        "Testing Records",
        result["test_rows"]
    )

    st.subheader("Model Performance Comparison")

    comparison = result["comparison"]

    st.dataframe(
        comparison,
        use_container_width=True
    )

    if not comparison.empty:

        metric_options = [
            column for column in comparison.columns
            if column != "Model"
        ]

        default_metric = (
            "RMSE" if result["task"] == "Regression" else "F1 Score"
        )

        if default_metric not in metric_options:
            default_metric = metric_options[0]

        selected_metric = st.selectbox(
            "Choose evaluation metric",
            options=metric_options,
            index=metric_options.index(default_metric)
        )

        fig = plot_model_comparison(
            comparison,
            selected_metric
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    st.subheader("Best Model Details")

    st.write(
        "Selected model:",
        result["best_model_name"]
    )

    st.write(
        "Target column:",
        result["target_column"]
    )

    st.info(
        "The best model is selected using the current test-set "
        "evaluation score. We will improve this selection process "
        "with validation and cross-validation during integration."
    )

    if result.get("failures"):

        with st.expander("Models that could not be trained"):

            st.dataframe(
                result["failures"],
                use_container_width=True
            )

    st.session_state["best_model"] = result["best_model"]
    st.session_state["best_model_name"] = result["best_model_name"]

else:
    st.info(
        "Configure your training options and click "
        "'Start Model Training' to begin."
    )


st.divider()

st.caption("ML Studio | Automated Model Training")
