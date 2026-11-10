# Monthly Car Sales Forecasting App

A Streamlit web app based on the `13_Monthly_Car_Sales_Forecasting_ML.ipynb` notebook.

## Features
- Loads the public monthly car sales dataset.
- Builds calendar, lag, rolling, moving-average, percentage-change and trend features.
- Trains Polynomial Regression, tuned XGBoost and tuned Random Forest models.
- Uses a chronological train/test split and compares MAE, MSE, RMSE and R².
- Selects the highest test-R² model and recursively forecasts 1–24 future months.
- Lets users download the forecast as CSV.

> The source dataset is a small historical series (1960–1968). This is an educational demonstration, not a current commercial car-market forecast. The notebook's `Accuracy (%) = R² × 100` is not classification accuracy.

## Run locally
1. Install Python 3.10 or 3.11.
2. Open a terminal in this project folder.
3. Create and activate a virtual environment (optional but recommended):
   - Windows: `py -m venv .venv` then `.venv\\Scripts\\activate`
   - macOS/Linux: `python3 -m venv .venv` then `source .venv/bin/activate`
4. Install packages: `pip install -r requirements.txt`
5. Run: `streamlit run app.py`
6. Open the local URL shown in the terminal (usually http://localhost:8501).

The first run trains and caches the models. It may take a little time.

## Push to GitHub
Create a **new empty repository** on GitHub (for example `monthly-car-sales-forecasting`).
Then run these commands from this folder:

```bash
git init
git add app.py requirements.txt README.md .gitignore
git commit -m "Build Streamlit car sales forecasting app"
git branch -M main
git remote add origin https://github.com/YOUR-USERNAME/monthly-car-sales-forecasting.git
git push -u origin main
```

Replace `YOUR-USERNAME` with your GitHub username. If Git says `remote origin already exists`, use:
`git remote set-url origin https://github.com/YOUR-USERNAME/monthly-car-sales-forecasting.git`

## Deploy on Streamlit Community Cloud
1. Push the project to GitHub.
2. Visit https://share.streamlit.io/ and sign in with GitHub.
3. Select **Create app**.
4. Choose your repository, branch `main`, and main file path `app.py`.
5. Click **Deploy** and wait for dependencies and model training to finish.
6. Copy the public app URL and add it to your GitHub repository's About section and README.

## Repository files
- `app.py` — Streamlit interface, feature engineering, model training and forecasting.
- `13_Monthly_Car_Sales_Forecasting_ML.ipynb` — original notebook used as the project reference.
- `requirements.txt` — Python dependencies.
- `README.md` — setup and deployment instructions.
