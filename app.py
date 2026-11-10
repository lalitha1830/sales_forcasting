
import streamlit as st
import pandas as pd

from core.data_loader import load_dataset, get_dataset_summary
from core.data_profiler import profile_dataset
from core.task_detector import recommend_tasks


# ---------------- PAGE CONFIGURATION ----------------

st.set_page_config(
    page_title="ML Studio | Universal AutoML",
    page_icon="🤖",
    layout="wide"
)


# ---------------- APPLICATION HEADER ----------------

st.title("🤖 ML Studio")
st.subheader("Universal Automated Machine Learning Platform")

st.markdown(
    """
    Upload your dataset, explore its structure, discover machine
    learning opportunities and build predictive models — all in one place.
    """
)

st.divider()


# ---------------- DATASET UPLOAD ----------------

st.header("📂 Upload Your Dataset")

uploaded_file = st.file_uploader(
    "Choose a CSV or Excel file",
    type=["csv", "xlsx", "xls"]
)


if uploaded_file is not None:

    try:
        df = load_dataset(uploaded_file)

        st.session_state["dataset"] = df
        st.session_state["dataset_name"] = uploaded_file.name

        st.success("Dataset uploaded successfully!")

    except Exception as error:
        st.error(str(error))


# ---------------- DATASET EXPLORATION ----------------

if "dataset" in st.session_state:

    df = st.session_state["dataset"]

    st.divider()

    st.header("📊 Dataset Overview")

    st.caption(
        f"Currently loaded: {st.session_state['dataset_name']}"
    )

    summary = get_dataset_summary(df)

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Rows", summary["rows"])
    col2.metric("Total Columns", summary["columns"])
    col3.metric("Missing Values", summary["missing_values"])
    col4.metric("Duplicate Rows", summary["duplicate_rows"])

    # Dataset preview

    st.subheader("Dataset Preview")

    st.dataframe(
        df.head(10),
        use_container_width=True
    )

    # Column information

    st.subheader("Column Information")

    column_info = pd.DataFrame({
        "Column": df.columns,
        "Data Type": df.dtypes.astype(str).values,
        "Missing Values": df.isnull().sum().values,
        "Unique Values": df.nunique().values
    })

    st.dataframe(
        column_info,
        use_container_width=True
    )

    # Detailed profiling

    with st.expander("Explore Detailed Dataset Profile"):

        try:
            profile = profile_dataset(df)
            st.write(profile)

        except Exception as error:
            st.warning(
                f"Detailed profiling could not be completed: {error}"
            )

    # Automatic task detection

    st.divider()

    st.header("🧠 AI-Assisted Task Recommendations")

    st.write(
        "ML Studio analyses your dataset and suggests suitable "
        "machine learning tasks."
    )

    try:
        recommendations = recommend_tasks(df)

        st.write(recommendations)

        st.info(
            "These recommendations use dataset characteristics "
            "and rule-based analysis. They are suggestions, not "
            "guaranteed AI decisions."
        )

    except Exception as error:
        st.warning(
            f"Task recommendations could not be generated: {error}"
        )

else:

    st.info(
        "👆 Upload a dataset to begin exploring ML Studio."
    )


# ---------------- FOOTER ----------------

st.divider()

st.caption(
    "ML Studio | Universal AutoML Platform | Built with Python and Streamlit"
)
