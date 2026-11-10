
import pandas as pd

from core.preprocessing import extract_datetime_features


def predict_new_data(
    model,
    new_data,
    expected_columns,
    target_column=None
):
    """
    Generate predictions using a previously trained model.

    Parameters:
        model: Trained sklearn pipeline.
        new_data: New uploaded pandas DataFrame.
        expected_columns: Original input columns used during training.
        target_column: Optional target column to exclude.

    Returns:
        DataFrame containing predictions.
    """

    if new_data is None or new_data.empty:
        raise ValueError("Please provide a valid dataset.")

    data = new_data.copy()

    # Remove the target column if it is included.
    if target_column and target_column in data.columns:
        data = data.drop(columns=[target_column])

    # Check required input columns.
    missing_columns = [
        col for col in expected_columns
        if col not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    # Keep only columns used during model training.
    data = data[expected_columns]

    # Apply the same date feature extraction.
    data = extract_datetime_features(data)

    try:
        predictions = model.predict(data)

    except Exception as error:
        raise ValueError(
            f"Prediction failed: {error}"
        )

    result = new_data.copy()

    result["Predicted_Value"] = predictions

    return result


def download_predictions(result):
    """
    Convert prediction results into downloadable CSV data.
    """

    if result is None or result.empty:
        raise ValueError("No prediction results available.")

    return result.to_csv(index=False).encode("utf-8")
