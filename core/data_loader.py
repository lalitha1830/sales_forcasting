import pandas as pd


def load_dataset(uploaded_file):
    """
    Load CSV or Excel datasets uploaded by users.
    """

    if uploaded_file is None:
        raise ValueError("Please upload a dataset.")

    file_name = uploaded_file.name.lower()

    try:
        if file_name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)

        elif file_name.endswith((".xlsx", ".xls")):
            df = pd.read_excel(uploaded_file)

        else:
            raise ValueError(
                "Unsupported file format. Upload CSV or Excel."
            )

        if df.empty:
            raise ValueError("The uploaded dataset is empty.")

        if len(df.columns) < 2:
            raise ValueError(
                "The dataset must contain at least two columns."
            )

        return df

    except Exception as error:
        raise ValueError(f"Dataset loading failed: {error}")


def get_dataset_summary(df):
    """
    Generate basic information about the dataset.
    """

    return {
        "rows": len(df),
        "columns": len(df.columns),
        "missing_values": int(df.isnull().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "column_names": list(df.columns),
        "data_types": df.dtypes.astype(str).to_dict()
    }
