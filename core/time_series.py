
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def prepare_time_series(df, date_column, target_column, frequency=None):
    """Clean and arrange time-series data in chronological order."""

    if date_column not in df.columns:
        raise ValueError("Date column not found.")

    if target_column not in df.columns:
        raise ValueError("Target column not found.")

    data = df[[date_column, target_column]].copy()

    data[date_column] = pd.to_datetime(
        data[date_column], errors="coerce"
    )
    data[target_column] = pd.to_numeric(
        data[target_column], errors="coerce"
    )

    data = data.dropna().sort_values(date_column)

    if data.empty:
        raise ValueError("No valid time-series data found.")

    # Combine records that have the same date.
    data = data.groupby(date_column)[target_column].mean()

    if frequency:
        data = data.resample(frequency).mean()
    elif len(data) >= 3:
        inferred_frequency = pd.infer_freq(data.index)
        if inferred_frequency is None:
            raise ValueError(
                "Could not determine the time frequency. "
                "Please provide one, such as D, W, or ME."
            )
        data = data.asfreq(inferred_frequency)
    else:
        raise ValueError("At least three dated records are required.")

    data = data.interpolate(method="time").ffill().bfill()

    if len(data) < 10:
        raise ValueError(
            "At least 10 time periods are required for forecasting."
        )

    return data


def create_time_features(series, lags=3):
    """Create lag and calendar features for forecasting."""

    data = pd.DataFrame({"value": series})

    for lag in range(1, lags + 1):
        data[f"lag_{lag}"] = data["value"].shift(lag)

    data["year"] = data.index.year
    data["month"] = data.index.month
    data["day"] = data.index.day
    data["day_of_week"] = data.index.dayofweek

    return data.dropna()


def forecast_time_series(
    df,
    date_column,
    target_column,
    periods=5,
    frequency=None,
    lags=3
):
    """Train, evaluate and generate future time-series forecasts."""

    if periods < 1:
        raise ValueError("Forecast periods must be at least 1.")

    if lags < 1:
        raise ValueError("Number of lags must be at least 1.")

    series = prepare_time_series(
        df, date_column, target_column, frequency
    )

    features = create_time_features(series, lags)

    if len(features) < 5:
        raise ValueError(
            "Not enough observations after creating lag features."
        )

    X = features.drop(columns=["value"])
    y = features["value"]

    # Chronological split: never shuffle time-series data.
    split_index = int(len(features) * 0.8)

    if split_index < 2 or split_index >= len(features):
        raise ValueError("Not enough data for training and testing.")

    X_train = X.iloc[:split_index]
    X_test = X.iloc[split_index:]
    y_train = y.iloc[:split_index]
    y_test = y.iloc[split_index:]

    model = RandomForestRegressor(
        n_estimators=100,
        random_state=42
    )

    model.fit(X_train, y_train)

    test_predictions = model.predict(X_test)

    mse = mean_squared_error(y_test, test_predictions)

    metrics = {
        "MAE": mean_absolute_error(y_test, test_predictions),
        "MSE": mse,
        "RMSE": np.sqrt(mse),
        "R2": r2_score(y_test, test_predictions)
    }

    # Retrain using all available observations.
    final_model = RandomForestRegressor(
        n_estimators=100,
        random_state=42
    )
    final_model.fit(X, y)

    # Generate future dates.
    future_dates = pd.date_range(
        start=series.index[-1],
        periods=periods + 1,
        freq=series.index.freq
    )[1:]

    history = list(series.values)
    future_values = []

    for future_date in future_dates:
        row = {}

        for lag in range(1, lags + 1):
            row[f"lag_{lag}"] = history[-lag]

        row["year"] = future_date.year
        row["month"] = future_date.month
        row["day"] = future_date.day
        row["day_of_week"] = future_date.dayofweek

        input_data = pd.DataFrame([row], columns=X.columns)
        prediction = float(final_model.predict(input_data)[0])

        future_values.append(prediction)
        history.append(prediction)

    forecast = pd.DataFrame({
        "Date": future_dates,
        "Forecast": future_values
    })

    return {
        "model": final_model,
        "metrics": metrics,
        "forecast": forecast,
        "historical_data": series,
        "test_actual": y_test,
        "test_predictions": test_predictions
    }
