import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

st.set_page_config(page_title="Monthly Car Sales Forecast", page_icon="🚗", layout="wide")

FEATURES = [
    "Year", "Month_Num", "Quarter", "Lag_1", "Lag_3", "Lag_6", "Lag_12",
    "Rolling_Mean_3", "Rolling_Std_3", "MA_6", "Pct_Change", "Trend"
]
DATA_URL = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/monthly-car-sales.csv"
FALLBACK_URL = "https://raw.githubusercontent.com/selva86/datasets/master/a10.csv"


@st.cache_data(show_spinner=False)
def load_data():
    errors = []
    for url in (DATA_URL, FALLBACK_URL):
        try:
            raw = pd.read_csv(url)
            raw.columns = [str(c).strip() for c in raw.columns]
            date_candidates = [c for c in raw.columns if c.lower() in ("month", "date", "time", "period", "ds")]
            date_col = date_candidates[0] if date_candidates else raw.columns[0]
            candidates = [c for c in raw.columns if c != date_col]
            sales_candidates = [c for c in candidates if any(k in c.lower() for k in ("sales", "value", "y", "count"))]
            sales_col = sales_candidates[0] if sales_candidates else candidates[0]
            data = raw[[date_col, sales_col]].copy()
            data.columns = ["Month", "Sales"]
            data["Month"] = pd.to_datetime(data["Month"], errors="coerce")
            data["Sales"] = pd.to_numeric(data["Sales"], errors="coerce")
            data = data.dropna(subset=["Month"]).drop_duplicates().sort_values("Month")
            data["Sales"] = data["Sales"].interpolate(method="linear").ffill().bfill()
            data = data.set_index("Month").asfreq("MS")
            data["Sales"] = data["Sales"].interpolate(method="linear").ffill().bfill()
            if len(data) >= 24 and data["Sales"].notna().all():
                data.index.name = "Date"
                return data, url
        except Exception as exc:
            errors.append(str(exc))
    raise RuntimeError("Could not load the public dataset. Check your internet connection. " + " | ".join(errors))


def make_features(data):
    frame = data.copy()
    frame["Year"] = frame.index.year
    frame["Month_Num"] = frame.index.month
    frame["Quarter"] = frame.index.quarter
    for lag in (1, 3, 6, 12):
        frame[f"Lag_{lag}"] = frame["Sales"].shift(lag)
    frame["Rolling_Mean_3"] = frame["Sales"].shift(1).rolling(3).mean()
    frame["Rolling_Std_3"] = frame["Sales"].shift(1).rolling(3).std()
    frame["MA_6"] = frame["Sales"].shift(1).rolling(6).mean()
    frame["Pct_Change"] = frame["Sales"].shift(1).pct_change() * 100
    frame["Trend"] = np.arange(len(frame))
    return frame.dropna()


def build_feature_row(target_date, history, last_trend, step):
    previous = history[-1]
    previous2 = history[-2]
    return {
        "Year": target_date.year,
        "Month_Num": target_date.month,
        "Quarter": target_date.quarter,
        "Lag_1": history[-1],
        "Lag_3": history[-3],
        "Lag_6": history[-6],
        "Lag_12": history[-12],
        "Rolling_Mean_3": np.mean(history[-3:]),
        "Rolling_Std_3": np.std(history[-3:]),
        "MA_6": np.mean(history[-6:]),
        "Pct_Change": ((previous - previous2) / previous2 * 100) if previous2 != 0 else 0.0,
        "Trend": last_trend + step,
    }


@st.cache_resource(show_spinner="Training and evaluating forecasting models…")
def train_models(data):
    model_data = make_features(data)
    X, y = model_data[FEATURES], model_data["Sales"]
    split = int(len(model_data) * 0.8)
    if split < 20 or len(model_data) - split < 5:
        raise ValueError("Not enough observations to create a reliable chronological train/test split.")
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    poly = Pipeline([
        ("scaler", StandardScaler()),
        ("poly", PolynomialFeatures(degree=2, include_bias=False)),
        ("linreg", LinearRegression())
    ])
    poly.fit(X_train, y_train)

    xgb_search = GridSearchCV(
        XGBRegressor(random_state=42, verbosity=0, objective="reg:squarederror"),
        {"n_estimators": [100, 200], "learning_rate": [0.05, 0.1], "max_depth": [2, 3]},
        cv=TimeSeriesSplit(n_splits=5), scoring="r2", n_jobs=-1
    )
    xgb_search.fit(X_train, y_train)

    rf_search = GridSearchCV(
        RandomForestRegressor(random_state=42),
        {"n_estimators": [100, 200], "max_depth": [4, 6], "min_samples_split": [2, 5]},
        cv=TimeSeriesSplit(n_splits=5), scoring="r2", n_jobs=-1
    )
    rf_search.fit(X_train, y_train)

    candidates = {
        "Polynomial Regression": poly,
        "XGBoost (Tuned)": xgb_search.best_estimator_,
        "Random Forest (Tuned)": rf_search.best_estimator_,
    }
    rows = []
    for name, model in candidates.items():
        pred = model.predict(X_test)
        mse = mean_squared_error(y_test, pred)
        rows.append({
            "Model": name,
            "MAE": mean_absolute_error(y_test, pred),
            "MSE": mse,
            "RMSE": np.sqrt(mse),
            "R² Score": r2_score(y_test, pred),
        })
    metrics = pd.DataFrame(rows).set_index("Model")
    best_name = metrics["R² Score"].idxmax()
    return candidates[best_name], best_name, metrics, y_test, candidates[best_name].predict(X_test), model_data


def make_forecast(data, model, months):
    history = list(data["Sales"].astype(float))
    last_date = data.index[-1]
    last_trend = len(data) - 1
    dates = pd.date_range(last_date + pd.DateOffset(months=1), periods=months, freq="MS")
    predictions = []
    for step, date in enumerate(dates, start=1):
        row = build_feature_row(date, history, last_trend, step)
        value = float(model.predict(pd.DataFrame([row])[FEATURES])[0])
        predictions.append(value)
        history.append(value)
    return pd.DataFrame({"Forecasted Sales": predictions}, index=dates)


st.title("🚗 Monthly Car Sales Forecasting")
st.caption("Machine Learning • Time-series feature engineering • Recursive monthly forecasting")
st.write("Explore historical monthly sales, compare regression models, and forecast future sales.")

try:
    with st.spinner("Loading monthly car sales dataset…"):
        data, source = load_data()
    st.success(f"Dataset loaded • {data.index.min():%b %Y} to {data.index.max():%b %Y} • {len(data)} monthly records")
    with st.spinner("Preparing models…"):
        best_model, best_name, metrics, y_test, test_predictions, model_data = train_models(data)
except Exception as exc:
    st.error(f"App setup failed: {exc}")
    st.info("Try refreshing the app. If it continues, check the dataset URL and dependency versions.")
    st.stop()

a, b, c = st.columns(3)
a.metric("Historical records", f"{len(data):,}")
b.metric("Selected model", best_name)
c.metric("Test R²", f"{metrics.loc[best_name, 'R² Score']:.3f}")

st.subheader("Historical sales")
fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(data.index, data["Sales"], linewidth=2)
ax.set_ylabel("Monthly sales")
ax.set_xlabel("Month")
ax.grid(alpha=0.25)
st.pyplot(fig)
plt.close(fig)

st.subheader("Model performance comparison")
st.dataframe(metrics.style.format("{:.3f}"), use_container_width=True)
st.caption("The notebook's 'Accuracy (%)' is R² × 100; it is not classification accuracy. R² can also be negative.")

st.subheader("Forecast future monthly sales")
months = st.slider("How many months should be forecast?", min_value=1, max_value=24, value=6)
forecast = make_forecast(data, best_model, months)
st.dataframe(forecast.style.format("{:,.0f}"), use_container_width=True)

fig2, ax2 = plt.subplots(figsize=(12, 5))
ax2.plot(data.index, data["Sales"], label="Historical sales", linewidth=2)
ax2.plot(forecast.index, forecast["Forecasted Sales"], label="Forecast", linestyle="--", marker="o")
ax2.axvline(data.index[-1], linestyle=":", alpha=0.7)
ax2.set_xlabel("Month")
ax2.set_ylabel("Sales")
ax2.set_title(f"Next {months} months • {best_name}")
ax2.legend()
ax2.grid(alpha=0.25)
st.pyplot(fig2)
plt.close(fig2)

csv = forecast.reset_index().rename(columns={"index": "Date"}).to_csv(index=False).encode("utf-8")
st.download_button("⬇️ Download forecast as CSV", data=csv, file_name="car_sales_forecast.csv", mime="text/csv")

with st.expander("About this project"):
    st.markdown("""
    - Features: calendar variables, 1/3/6/12-month lags, rolling statistics, moving average, percentage change, and trend.
    - Models: Polynomial Regression, tuned XGBoost Regressor, and tuned Random Forest Regressor.
    - Evaluation: chronological 80/20 train-test split; MAE, MSE, RMSE, and R².
    - Forecasting: recursive multi-step prediction, feeding each prediction into the next month's features.
    - Dataset: public monthly car sales dataset (1960–1968). This small historical dataset is for demonstration and education, not a current market forecast.
    """)
