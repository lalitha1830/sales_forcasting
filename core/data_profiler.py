
import pandas as pd


def profile_dataset(df):
    """
    Analyse a dataset and return its basic characteristics.
    """

    if df is None or df.empty:
        raise ValueError("Dataset is empty.")

    total_rows, total_columns = df.shape

    numeric_columns = df.select_dtypes(
        include=["number"]
    ).columns.tolist()

    categorical_columns = df.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()

    datetime_columns = []
    for column in df.columns:
        if column in numeric_columns:
            continue

        # Check whether the column name suggests a date.
        name = str(column).lower()
        date_keywords = [
            "date", "time", "day", "month", "year",
            "timestamp"
        ]

        if any(word in name for word in date_keywords):
            parsed = pd.to_datetime(
                df[column], errors="coerce"
            )
            if parsed.notna().mean() >= 0.7:
                datetime_columns.append(column)

    missing_values = df.isnull().sum().to_dict()

    missing_percentages = (
        (df.isnull().mean() * 100).round(2).to_dict()
    )

    duplicate_rows = int(df.duplicated().sum())

    unique_values = {
        column: int(df[column].nunique())
        for column in df.columns
    }

    numeric_summary = {}
    if numeric_columns:
        numeric_summary = (
            df[numeric_columns]
            .describe()
            .round(2)
            .to_dict()
        )

    categorical_summary = {
        column: df[column].nunique()
        for column in categorical_columns
    }

    return {
        "total_rows": total_rows,
        "total_columns": total_columns,
        "numeric_columns": numeric_columns,
        "categorical_columns": categorical_columns,
        "datetime_columns": datetime_columns,
        "missing_values": missing_values,
        "missing_percentages": missing_percentages,
        "duplicate_rows": duplicate_rows,
        "unique_values": unique_values,
        "numeric_summary": numeric_summary,
        "categorical_summary": categorical_summary
    }
