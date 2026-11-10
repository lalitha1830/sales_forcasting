# ForecastLab — Dynamic Time-Series Forecasting Platform

A Streamlit application that accepts user-uploaded CSV and Excel datasets, prepares a regular daily, weekly, or monthly time series, compares three machine-learning regressors, and generates multi-step forecasts.

## Features

- Use the original public monthly car-sales dataset or upload a CSV / Excel workbook.
- Choose the date column and numerical target column.
- Select daily, weekly, or monthly frequency.
- Combine repeated dates using mean or sum.
- Preview raw and cleaned data.
- Handle invalid values, duplicate dates, missing intervals, and numeric text.
- Train Polynomial Regression, tuned XGBoost, and tuned Random Forest models.
- Use a chronological 80/20 train/test split and TimeSeriesSplit cross-validation for tree-model tuning.
- Compare MAE, MSE, RMSE, and R².
- Select the best model by lowest test RMSE.
- Generate recursive forecasts and download results as CSV.
- Light professional interface.

## Dataset requirements

The uploaded file must contain:
- One date/time column.
- One numerical column to forecast.
- At least 36 observations after conversion to the selected frequency.

The application aggregates duplicate dates using the selected mean/sum setting and fills missing time intervals by linear interpolation. For meaningful forecasting, use a dataset with a consistent time interval and enough history. Daily, weekly, and monthly forecasting are supported; this is not a general-purpose model for unrelated, non-time-series tables.

## Run locally

Use Python 3.10 or newer.

```bash
python -m venv .venv
```

Windows:
```bash
.venv\Scripts\activate
```

macOS/Linux:
```bash
source .venv/bin/activate
```

Install dependencies and run:
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Community Cloud

1. Push `app.py`, `requirements.txt`, and `README.md` to a GitHub repository.
2. Open https://share.streamlit.io/ and sign in with GitHub.
3. Select **Create app**.
4. Choose your repository and `main` branch.
5. Set the main file path to `app.py`.
6. Select **Deploy**.

## Notes

- The sample dataset is the historical monthly car-sales series (1960–1968), included as a demonstration.
- Forecasts are estimates and depend on the quality, length, and regularity of the uploaded data.
- Model tuning can take a few minutes, especially for larger datasets.
- This project is educational and should not be treated as a guarantee of future business outcomes.
