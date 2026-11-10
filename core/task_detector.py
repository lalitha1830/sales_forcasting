
import pandas as pd


def find_datetime_columns(df):
    """Identify columns that appear to contain dates."""

    date_columns = []

    keywords = [
        "date", "time", "timestamp",
        "datetime", "day"
    ]

    for column in df.columns:
        name = str(column).lower()

        if not any(word in name for word in keywords):
            continue

        parsed = pd.to_datetime(
            df[column], errors="coerce"
        )

        if parsed.notna().mean() >= 0.7:
            date_columns.append(column)

    return date_columns


def get_target_suggestions(df):
    """Suggest possible target columns."""

    suggestions = []

    for column in df.columns:

        name = str(column).lower()

        if any(word in name for word in [
            "id", "index", "serial", "number"
        ]) and df[column].nunique() > 0.8 * len(df):
            continue

        unique_count = df[column].nunique()
        dtype = df[column].dtype

        if unique_count <= 1:
            continue

        if pd.api.types.is_numeric_dtype(dtype):
            if unique_count <= min(20, max(2, len(df) * 0.05)):
                task = "Classification"
            else:
                task = "Regression"

        else:
            task = "Classification"

        suggestions.append({
            "column": column,
            "suggested_task": task,
            "unique_values": int(unique_count)
        })

    return suggestions


def recommend_tasks(df):
    """
    Recommend machine learning tasks based on
    dataset structure.
    """

    if df is None or df.empty:
        raise ValueError("Please provide a valid dataset.")

    date_columns = find_datetime_columns(df)

    numeric_columns = df.select_dtypes(
        include=["number"]
    ).columns.tolist()

    categorical_columns = df.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()

    target_suggestions = get_target_suggestions(df)

    recommendations = []

    # Regression
    if any(
        item["suggested_task"] == "Regression"
        for item in target_suggestions
    ):
        recommendations.append({
            "task": "Regression",
            "reason": "The dataset contains numerical columns "
                      "that may be suitable for predicting "
                      "continuous values."
        })

    # Classification
    if any(
        item["suggested_task"] == "Classification"
        for item in target_suggestions
    ):
        recommendations.append({
            "task": "Classification",
            "reason": "The dataset contains columns with "
                      "categories or a limited number of "
                      "unique values."
        })

    # Clustering
    if len(df.columns) >= 2:
        recommendations.append({
            "task": "Clustering",
            "reason": "The dataset has multiple features "
                      "that may help discover natural groups."
        })

    # Time-series forecasting
    if date_columns and numeric_columns:
        recommendations.append({
            "task": "Time Series Forecasting",
            "reason": "Date/time columns and numerical "
                      "values were detected."
        })

    # Anomaly detection
    if len(numeric_columns) >= 1:
        recommendations.append({
            "task": "Anomaly Detection",
            "reason": "Numerical features are available "
                      "for identifying unusual observations."
        })

    # Dimensionality reduction
    if len(numeric_columns) >= 2:
        recommendations.append({
            "task": "Dimensionality Reduction",
            "reason": "Multiple numerical features are "
                      "available for feature compression "
                      "and visualization."
        })

    return {
        "recommended_tasks": recommendations,
        "date_columns": date_columns,
        "numeric_columns": numeric_columns,
        "categorical_columns": categorical_columns,
        "target_suggestions": target_suggestions
    }
