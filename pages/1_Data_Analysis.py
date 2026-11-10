
import streamlit as st
import pandas as pd

from utils.visualizations import (
    plot_dataset_distribution,
    plot_missing_values,
    plot_correlation_heatmap
)


st.set_page_config(
    page_title="Data Analysis | ML Studio",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Data Analysis")
st.write("Explore, understand and visualize your dataset.")

if "dataset" not in st.session_state:
    st.warning("Please upload a dataset from the ML Studio homepage first.")
    st.stop()

df = st.session_state["dataset"]

st.caption(
    f"Dataset: {st.session_state.get('dataset_name', 'Uploaded dataset')}"
)

# ---------------- BASIC STATISTICS ----------------

st.header("Dataset Statistics")

col1, col2, col3 = st.columns(3)

col1.metric("Rows", len(df))
col2.metric("Columns", len(df.columns))
col3.metric("Total Missing Values", int(df.isnull().sum().sum()))

st.divider()

# ---------------- COLUMN ANALYSIS ----------------

st.header("Column Analysis")

numeric_columns = df.select_dtypes(
    include="number"
).columns.tolist()

categorical_columns = df.select_dtypes(
    include=["object", "category", "bool"]
).columns.tolist()

selected_column = st.selectbox(
    "Choose a column to analyse",
    options=df.columns.tolist()
)

if selected_column:

    st.subheader(f"Analysis: {selected_column}")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Unique Values",
        df[selected_column].nunique()
    )

    col2.metric(
        "Missing Values",
        int(df[selected_column].isnull().sum())
    )

    col3.metric(
        "Data Type",
        str(df[selected_column].dtype)
    )

    try:
        fig = plot_dataset_distribution(
            df,
            selected_column
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    except Exception as error:
        st.error(f"Could not create chart: {error}")

# ---------------- NUMERICAL ANALYSIS ----------------

st.divider()

st.header("Numerical Feature Analysis")

if numeric_columns:

    st.write("Statistical summary of numerical columns.")

    st.dataframe(
        df[numeric_columns].describe(),
        use_container_width=True
    )

else:
    st.info("No numerical columns found.")

# ---------------- MISSING VALUE ANALYSIS ----------------

st.divider()

st.header("Missing Value Analysis")

missing_fig = plot_missing_values(df)

if missing_fig is not None:

    st.plotly_chart(
        missing_fig,
        use_container_width=True
    )

else:
    st.success("No missing values found in this dataset!")

# ---------------- CORRELATION ANALYSIS ----------------

st.divider()

st.header("Feature Correlation")

correlation_fig = plot_correlation_heatmap(df)

if correlation_fig is not None:

    st.plotly_chart(
        correlation_fig,
        use_container_width=True
    )

    st.caption(
        "Correlation measures the strength and direction of "
        "linear relationships between numerical features. "
        "Correlation does not prove causation."
    )

else:
    st.info(
        "At least two numerical columns are required "
        "to generate a correlation heatmap."
    )

# ---------------- FOOTER ----------------

st.divider()

st.caption("ML Studio | Data Analysis Module")
