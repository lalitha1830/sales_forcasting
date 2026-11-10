import warnings
warnings.filterwarnings("ignore")

from io import BytesIO
import hashlib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor


# ----------------------------- PAGE & THEME -----------------------------
st.set_page_config(
    page_title="ForecastLab | Time-Series Forecasting",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    :root { color-scheme: light; }
    .stApp { background: #F5F8FC; color: #172B4D; }
    [data-testid="stHeader"] { background: rgba(245,248,252,0.92); }
    [data-testid="stSidebar"] { background: #FFFFFF; border-right: 1px solid #E5EAF2; }
    [data-testid="stSidebar"] * { color: #243B53; }
    h1, h2, h3 { color: #17365D !important; }
    p, label, span { color: #334E68; }
    div[data-testid="stMetric"] {
        background: #FFFFFF; border: 1px solid #E4EAF2;
        padding: 16px; border-radius: 14px;
        box-shadow: 0 2px 8px rgba(20,50,90,.04);
    }
    div.stButton > button, div.stDownloadButton > button {
        background: #176B87; color: #FFFFFF; border: 0;
        border-radius: 9px; padding: .55rem 1rem; font-weight: 600;
    }
    div.stButton > button:hover, div.stDownloadButton > button:hover {
        background: #12566D; color: #FFFFFF; border: 0;
    }
    [data-testid="stFileUploader"] {
        background: #FFFFFF; border: 1px dashed #9FB3C8;
        border-radius: 12px; padding: 10px;
    }
    [data-testid="stDataFrame"] { background: #FFFFFF; }
    .hero {
        padding: 1.35rem 1.5rem; border-radius: 18px;
        background: linear-gradient(115deg,#E5F4F8,#FFFFFF 65%);
        border: 1px solid #D5E8EF; margin-bottom: 1rem;
    }
    .hero p { margin-bottom: 0; color: #486581; }
    .small-note { color: #627D98; font-size: .88rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ----------------------------- DATA FUNCTIONS -----------------------------
DEFAULT_DATA_URL = (
    "https://raw.githubusercontent.com/jbrownlee/Datasets/master/monthly-car-sales.csv"
)


@st.cache_data(show_spinner=False)
def load_default_dataset():
    """Load the original monthly car-sales dataset, with a public fallback."""
    try:
        raw = pd.read_csv(DEFAULT_DATA_URL)
        return raw, DEFAULT_DATA_URL
    except Exception as exc:
        raise RuntimeError(
            "Could not load the original monthly car-sales dataset. "
            "Check your internet connection or upload your own CSV/Excel file. "
            f"Details: {exc}"
        )


def read_uploaded_dataset(uploaded_file):
    """Read CSV or Excel data uploaded by the user."""
    file_bytes = uploaded_file.getvalue()
    if uploaded_file.name.lower().endswith(".csv"):
        return pd.read_csv(BytesIO(file_bytes))
    if uploaded_file.name.lower().endswith((".xlsx", ".xls")):
        return pd.read_excel(BytesIO(file_bytes))
    raise ValueError("Please upload a CSV or Excel file.")


def make_regular_series(raw, date_col, target_col, frequency, aggregation):
    """Clean and resample the selected date and numeric target columns."""
    df = raw[[date_col, target_col]].copy()
    df.columns = ["Date", "Value"]
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Value"] = pd.to_numeric(
        df["Value"].astype(str).str.replace(",", "", regex=False), errors="coerce"
    )
    df = df.dropna(subset=["Date", "Value"])

    if df.empty:
        raise ValueError("No valid date and numeric-value rows were found.")

    # Do not manufacture high-frequency observations from lower-frequency data.
    median_gap_days = (
        df["Date"].drop_duplicates().sort_values().diff().dt.total_seconds().div(86400).median()
    )
    if pd.notna(median_gap_days):
        if frequency == "Daily" and median_gap_days > 1.5:
            raise ValueError(
                "This dataset is not daily. Choosing Daily would invent interpolated "
                "values between observations. Choose Weekly or Monthly instead."
            )
        if frequency == "Weekly" and median_gap_days > 10:
            raise ValueError(
                "This dataset is less frequent than weekly. Choose Monthly, or upload "
                "weekly/daily observations."
            )
        if frequency == "Monthly" and median_gap_days > 45:
            raise ValueError(
                "This dataset appears less frequent than monthly. Upload a more detailed "
                "time series or choose an appropriate interval."
            )

    # Remove timezone information and normalize dates to avoid duplicate timestamps.
    try:
        if getattr(df["Date"].dt, "tz", None) is not None:
            df["Date"] = df["Date"].dt.tz_localize(None)
    except Exception:
        pass

    if frequency == "Daily":
        rule = "D"
        df["Date"] = df["Date"].dt.normalize()
    elif frequency == "Weekly":
        rule = "W-MON"
    else:
        rule = "MS"
        df["Date"] = df["Date"].dt.to_period("M").dt.to_timestamp()

    grouped = df.groupby("Date")["Value"]
    if aggregation == "Sum":
        series = grouped.sum()
    else:
        series = grouped.mean()

    series = series.sort_index()
    series = series.resample(rule).mean()

    # Fill missing dates so the series has a regular interval.
    series = series.interpolate(method="linear").ffill().bfill()
    result = series.to_frame(name="Value").dropna()

    if len(result) < 36:
        raise ValueError(
            f"Only {len(result)} {frequency.lower()} observations remain after cleaning. "
            "Please upload at least 36 observations for this forecasting workflow."
        )
    result.index.name = "Date"
    return result


# ----------------------------- FEATURE ENGINEERING -----------------------------
FEATURES = [
    "Year", "Month", "Quarter", "DayOfWeek", "DayOfYear", "WeekOfYear",
    "Season_Sin", "Season_Cos", "Trend",
    "Lag_1", "Lag_2", "Lag_3", "Lag_6", "Lag_12",
    "Rolling_Mean_3", "Rolling_Mean_6", "Rolling_Std_3", "Pct_Change",
]


def date_features(date, frequency, trend):
    if frequency == "Daily":
        season_position, season_length = date.dayofweek, 7
    elif frequency == "Weekly":
        season_position, season_length = int(date.isocalendar().week), 52
    else:
        season_position, season_length = date.month - 1, 12

    angle = 2 * np.pi * season_position / season_length
    return {
        "Year": date.year,
        "Month": date.month,
        "Quarter": date.quarter,
        "DayOfWeek": date.dayofweek,
        "DayOfYear": date.dayofyear,
        "WeekOfYear": int(date.isocalendar().week),
        "Season_Sin": np.sin(angle),
        "Season_Cos": np.cos(angle),
        "Trend": trend,
    }


def build_training_frame(series, frequency):
    values = series["Value"].astype(float)
    frame = pd.DataFrame(index=series.index)
    for col, value in date_features(series.index[0], frequency, 0).items():
        frame[col] = [date_features(d, frequency, i)[col] for i, d in enumerate(series.index)]

    for lag in (1, 2, 3, 6, 12):
        frame[f"Lag_{lag}"] = values.shift(lag)

    frame["Rolling_Mean_3"] = values.shift(1).rolling(3).mean()
    frame["Rolling_Mean_6"] = values.shift(1).rolling(6).mean()
    frame["Rolling_Std_3"] = values.shift(1).rolling(3).std()
    frame["Pct_Change"] = values.shift(1).pct_change() * 100
    frame["Target"] = values
    frame = frame.replace([np.inf, -np.inf], np.nan).dropna()
    return frame


def build_future_row(future_date, history, frequency, trend):
    row = date_features(future_date, frequency, trend)
    row.update({
        "Lag_1": history[-1],
        "Lag_2": history[-2],
        "Lag_3": history[-3],
        "Lag_6": history[-6],
        "Lag_12": history[-12],
        "Rolling_Mean_3": float(np.mean(history[-3:])),
        "Rolling_Mean_6": float(np.mean(history[-6:])),
        "Rolling_Std_3": float(np.std(history[-3:], ddof=1)),
        "Pct_Change": (
            (history[-1] - history[-2]) / history[-2] * 100
            if history[-2] != 0 else 0.0
        ),
    })
    return pd.DataFrame([row])[FEATURES]


def next_dates(last_date, periods, frequency):
    if frequency == "Daily":
        return pd.date_range(last_date + pd.Timedelta(days=1), periods=periods, freq="D")
    if frequency == "Weekly":
        return pd.date_range(last_date + pd.Timedelta(days=7), periods=periods, freq="W-MON")
    return pd.date_range(last_date + pd.offsets.MonthBegin(1), periods=periods, freq="MS")


# ----------------------------- MODEL TRAINING -----------------------------
def train_and_evaluate(series, frequency):
    model_frame = build_training_frame(series, frequency)
    if len(model_frame) < 20:
        raise ValueError(
            "Not enough usable rows after creating lag features. Upload a longer time series."
        )

    X = model_frame[FEATURES]
    y = model_frame["Target"]
    split = int(len(model_frame) * 0.8)
    if split < 12 or len(model_frame) - split < 4:
        raise ValueError("Not enough rows for a chronological train/test split.")

    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    polynomial = Pipeline([
        ("scaler", StandardScaler()),
        ("polynomial", PolynomialFeatures(degree=2, include_bias=False)),
        ("regression", LinearRegression()),
    ])
    polynomial.fit(X_train, y_train)

    n_splits = min(3, max(2, len(X_train) // 5))
    time_cv = TimeSeriesSplit(n_splits=n_splits)

    xgb_search = GridSearchCV(
        XGBRegressor(
            objective="reg:squarederror", random_state=42,
            verbosity=0, n_jobs=1
        ),
        {
            "n_estimators": [100, 200],
            "learning_rate": [0.05, 0.1],
            "max_depth": [2, 3],
        },
        cv=time_cv,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
    )
    xgb_search.fit(X_train, y_train)

    rf_search = GridSearchCV(
        RandomForestRegressor(random_state=42, n_jobs=-1),
        {
            "n_estimators": [100, 200],
            "max_depth": [4, 6],
            "min_samples_split": [2, 5],
        },
        cv=time_cv,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
    )
    rf_search.fit(X_train, y_train)

    models = {
        "Polynomial Regression": polynomial,
        "XGBoost (Tuned)": xgb_search.best_estimator_,
        "Random Forest (Tuned)": rf_search.best_estimator_,
    }

    metric_rows = []
    for name, model in models.items():
        predictions = model.predict(X_test)
        mse = mean_squared_error(y_test, predictions)
        metric_rows.append({
            "Model": name,
            "MAE": mean_absolute_error(y_test, predictions),
            "MSE": mse,
            "RMSE": np.sqrt(mse),
            "R² Score": r2_score(y_test, predictions),
        })

    metrics = pd.DataFrame(metric_rows).set_index("Model").sort_values("RMSE")
    best_name = metrics.index[0]
    return {
        "models": models,
        "best_model": models[best_name],
        "best_name": best_name,
        "metrics": metrics,
        "y_test": y_test,
        "test_predictions": models[best_name].predict(X_test),
        "model_frame": model_frame,
        "frequency": frequency,
    }


def create_forecast(series, model, frequency, periods):
    history = list(series["Value"].astype(float))
    dates = next_dates(series.index[-1], periods, frequency)
    predictions = []
    for step, future_date in enumerate(dates, start=1):
        future_X = build_future_row(future_date, history, frequency, len(history))
        prediction = float(model.predict(future_X)[0])
        predictions.append(prediction)
        history.append(prediction)
    return pd.DataFrame({"Forecast": predictions}, index=dates).rename_axis("Date")


# ----------------------------- SIDEBAR -----------------------------
st.sidebar.title("📈 ForecastLab")
st.sidebar.caption("Time-Series Forecasting Platform")
page = st.sidebar.radio(
    "NAVIGATION",
    ["Overview", "Data & Upload", "Model Training", "Forecast & Download"],
)
st.sidebar.divider()
st.sidebar.markdown("**Supported formats**")
st.sidebar.caption("CSV • Excel (.xlsx, .xls)")
st.sidebar.markdown("**Forecast intervals**")
st.sidebar.caption("Daily • Weekly • Monthly")


# ----------------------------- HEADER -----------------------------
st.markdown(
    """
    <div class="hero">
      <h1 style="margin:0;color:#17365D;">ForecastLab</h1>
      <p>Upload a time-series dataset, compare machine-learning models, and forecast future values.</p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ----------------------------- DATA SOURCE & CONFIG -----------------------------
with st.sidebar:
    st.subheader("Dataset")
    source_choice = st.radio("Choose data source", ["Use sample car-sales data", "Upload my dataset"])
    uploaded_file = None
    if source_choice == "Upload my dataset":
        uploaded_file = st.file_uploader(
            "Upload CSV or Excel",
            type=["csv", "xlsx", "xls"],
            help="Your file should contain one date column and one numeric column.",
        )

if source_choice == "Upload my dataset" and uploaded_file is None:
    st.info("Upload a CSV or Excel file in the left sidebar to get started.")
    st.markdown("### Example dataset format")
    st.dataframe(pd.DataFrame({
        "Date": pd.date_range("2023-01-01", periods=5, freq="MS"),
        "Sales": [120, 135, 128, 150, 162],
    }), use_container_width=True)
    st.stop()

try:
    if source_choice == "Use sample car-sales data":
        raw_df, data_origin = load_default_dataset()
        st.sidebar.caption("Sample source: public monthly car-sales data")
    else:
        raw_df = read_uploaded_dataset(uploaded_file)
        data_origin = uploaded_file.name
except Exception as exc:
    st.error(f"Could not read the dataset: {exc}")
    st.stop()

if raw_df.empty or len(raw_df.columns) < 2:
    st.error("Your file must contain at least two columns: a date column and a numeric value column.")
    st.stop()

st.sidebar.subheader("Forecast settings")
date_guess = next((c for c in raw_df.columns if str(c).strip().lower() in
                   ["date", "month", "time", "period", "ds"]), raw_df.columns[0])
date_default = list(raw_df.columns).index(date_guess)
date_col = st.sidebar.selectbox("Date column", list(raw_df.columns), index=date_default)

numeric_columns = [
    c for c in raw_df.columns
    if c != date_col and pd.to_numeric(
        raw_df[c].astype(str).str.replace(",", "", regex=False), errors="coerce"
    ).notna().sum() > 0
]
if not numeric_columns:
    st.error("No numeric target column was found. Your dataset needs a numerical column to forecast.")
    st.stop()

target_guess = next((c for c in numeric_columns if any(
    word in str(c).lower() for word in ["sales", "value", "revenue", "quantity", "demand"]
)), numeric_columns[0])
target_col = st.sidebar.selectbox(
    "Value to forecast",
    numeric_columns,
    index=numeric_columns.index(target_guess),
)
frequency = st.sidebar.selectbox("Forecast frequency", ["Daily", "Weekly", "Monthly"], index=2)
aggregation = st.sidebar.selectbox(
    "Combine repeated dates using",
    ["Mean", "Sum"],
    help="Mean is useful for measurements; Sum is useful for sales, orders or revenue totals.",
)

try:
    series = make_regular_series(raw_df, date_col, target_col, frequency, aggregation)
except Exception as exc:
    st.error(f"Dataset validation: {exc}")
    st.stop()

data_signature = hashlib.md5(
    (str(data_origin) + str(date_col) + str(target_col) + frequency + aggregation
     + raw_df.to_csv(index=False)).encode()
).hexdigest()

if st.session_state.get("data_signature") != data_signature:
    st.session_state["data_signature"] = data_signature
    st.session_state.pop("trained_result", None)
    st.session_state.pop("forecast_result", None)


# ----------------------------- PAGE: OVERVIEW -----------------------------
if page == "Overview":
    st.subheader("Welcome to your forecasting workspace")
    st.write(
        "ForecastLab turns a dated numerical dataset into a time-series forecasting workflow. "
        "Start with the sample car-sales dataset or upload your own data."
    )
    m1, m2, m3 = st.columns(3)
    m1.metric("Observations after cleaning", f"{len(series):,}")
    m2.metric("Forecast interval", frequency)
    m3.metric("Selected target", str(target_col))
    st.markdown("### Historical trend")
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(series.index, series["Value"], linewidth=2, color="#176B87")
    ax.set_xlabel("Date")
    ax.set_ylabel(str(target_col))
    ax.grid(alpha=0.2)
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)
    st.markdown("### How to use the application")
    st.markdown(
        """
        1. **Data & Upload:** choose the sample dataset or upload a CSV/Excel file.
        2. **Model Training:** train and compare the three regression algorithms.
        3. **Forecast & Download:** choose the forecast horizon and download the predictions.
        """
    )


# ----------------------------- PAGE: DATA -----------------------------
elif page == "Data & Upload":
    st.subheader("Dataset preview and data quality")
    st.caption(f"Current source: {data_origin}")
    a, b, c = st.columns(3)
    a.metric("Rows in uploaded file", f"{len(raw_df):,}")
    b.metric("Rows after processing", f"{len(series):,}")
    c.metric("Missing values after processing", f"{int(series['Value'].isna().sum())}")
    st.markdown("#### Original data preview")
    st.dataframe(raw_df.head(20), use_container_width=True)
    st.markdown("#### Processed time series")
    st.dataframe(series.head(20), use_container_width=True)
    st.markdown("#### Processed data trend")
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(series.index, series["Value"], color="#176B87", linewidth=2)
    ax.set_xlabel("Date")
    ax.set_ylabel(str(target_col))
    ax.grid(alpha=0.2)
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)
    st.download_button(
        "Download cleaned dataset",
        data=series.reset_index().to_csv(index=False).encode("utf-8"),
        file_name="cleaned_time_series.csv",
        mime="text/csv",
    )


# ----------------------------- PAGE: TRAINING -----------------------------
elif page == "Model Training":
    st.subheader("Train and compare machine-learning models")
    st.write(
        "The data is split chronologically: earlier observations train the models and later "
        "observations test them. Hyperparameters for XGBoost and Random Forest are tuned with "
        "TimeSeriesSplit cross-validation."
    )
    st.info(
        f"Current dataset: {len(series)} {frequency.lower()} observations. "
        "Training may take a few minutes because model tuning is performed."
    )
    if st.button("🚀 Train all models", type="primary"):
        try:
            with st.spinner("Training Polynomial Regression, XGBoost and Random Forest..."):
                result = train_and_evaluate(series, frequency)
            st.session_state["trained_result"] = result
            st.session_state.pop("forecast_result", None)
            st.success(f"Training complete. Best model by lowest test RMSE: {result['best_name']}")
        except Exception as exc:
            st.error(f"Training failed: {exc}")

    result = st.session_state.get("trained_result")
    if result is not None:
        st.markdown("### Model performance")
        st.dataframe(result["metrics"].style.format("{:,.3f}"), use_container_width=True)
        best = result["best_name"]
        x, y, z = st.columns(3)
        x.metric("Selected model", best)
        y.metric("Test RMSE", f"{result['metrics'].loc[best, 'RMSE']:,.3f}")
        z.metric("Test R²", f"{result['metrics'].loc[best, 'R² Score']:.3f}")
        st.caption("Lower MAE, MSE and RMSE indicate smaller prediction errors. R² can be negative.")
        fig, ax = plt.subplots(figsize=(11, 4))
        ax.plot(result["y_test"].index, result["y_test"], label="Actual", color="#176B87", linewidth=2)
        ax.plot(result["y_test"].index, result["test_predictions"], label="Predicted", color="#E07A5F", linestyle="--")
        ax.set_title("Actual vs predicted values on the test period")
        ax.legend()
        ax.grid(alpha=0.2)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
    else:
        st.warning("No trained models yet. Click **Train all models** to begin.")


# ----------------------------- PAGE: FORECAST -----------------------------
elif page == "Forecast & Download":
    st.subheader("Generate future predictions")
    result = st.session_state.get("trained_result")
    if result is None:
        st.warning("Please open Model Training and train the models first.")
        st.stop()

    max_periods = 90 if frequency == "Daily" else (52 if frequency == "Weekly" else 36)
    default_periods = 7 if frequency == "Daily" else (8 if frequency == "Weekly" else 6)
    periods = st.slider(
        f"Number of future {frequency.lower()} periods",
        min_value=1,
        max_value=max_periods,
        value=default_periods,
    )
    if st.button("Generate forecast", type="primary"):
        with st.spinner("Generating future predictions..."):
            forecast_df = create_forecast(
                series, result["best_model"], frequency, periods
            )
        st.session_state["forecast_result"] = forecast_df
        st.success("Forecast generated successfully.")

    forecast_df = st.session_state.get("forecast_result")
    if forecast_df is not None:
        st.markdown(f"### Forecast using {result['best_name']}")
        st.dataframe(forecast_df.style.format("{:,.2f}"), use_container_width=True)
        fig, ax = plt.subplots(figsize=(11, 5))
        ax.plot(series.index, series["Value"], label="Historical", color="#176B87", linewidth=2)
        ax.plot(forecast_df.index, forecast_df["Forecast"], label="Forecast", color="#E07A5F", linestyle="--", marker="o")
        ax.axvline(series.index[-1], color="#829AB1", linestyle=":")
        ax.set_xlabel("Date")
        ax.set_ylabel(str(target_col))
        ax.legend()
        ax.grid(alpha=0.2)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        csv_bytes = forecast_df.reset_index().to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇ Download forecast as CSV",
            data=csv_bytes,
            file_name="time_series_forecast.csv",
            mime="text/csv",
        )

st.sidebar.divider()
st.sidebar.caption("ForecastLab • Educational machine-learning application")
st.sidebar.caption("Forecasts are estimates, not guarantees.")
