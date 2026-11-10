
# 🤖 ML Studio — Universal AutoML Platform

### Machine Learning Made Simple, Accessible and Automated

ML Studio is a universal automated machine learning platform designed to help users analyze structured datasets, build machine learning models, evaluate performance and generate predictions without requiring advanced machine learning expertise.

The platform aims to support multiple machine learning tasks through a simple and interactive interface.

## ✨ Key Features

### 1. Dataset Management
- Upload CSV and Excel datasets.
- Preview uploaded data.
- Identify missing values and duplicate records.
- Explore numerical and categorical features.

### 2. Automated Data Analysis
- Dataset profiling.
- Statistical summaries.
- Missing-value visualization.
- Correlation heatmaps.
- Interactive data distributions.

### 3. Machine Learning
- Regression.
- Classification.
- Clustering.
- Anomaly detection.
- Dimensionality reduction.
- Time-series forecasting.

### 4. Automated Model Selection
- Access multiple machine learning algorithms.
- Compare model performance.
- Identify the best-performing model.
- Hyperparameter tuning.

### 5. Model Evaluation
- Accuracy, Precision, Recall and F1-score.
- MAE, MSE, RMSE and R².
- Clustering evaluation.
- Anomaly detection results.

### 6. Explainable AI
- Permutation-based feature importance.
- Understand which features influence model performance.

### 7. Predictions
- Generate predictions on new datasets.
- View prediction results.
- Export predictions as CSV files.

## 🛠️ Technology Stack

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Frontend | Streamlit |
| Data Processing | Pandas, NumPy |
| Machine Learning | Scikit-learn |
| Forecasting | Statsmodels |
| Visualization | Plotly |
| Hyperparameter Tuning | GridSearchCV |
| Explainable AI | Permutation Importance |
| Version Control | Git and GitHub |

## 📁 Project Structure

```text
sales_forcasting/
│
├── app.py
├── requirements.txt
├── README.md
│
├── core/
│   ├── data_loader.py
│   ├── data_profiler.py
│   ├── preprocessing.py
│   ├── task_detector.py
│   ├── model_registry.py
│   ├── auto_ml.py
│   ├── evaluation.py
│   ├── prediction.py
│   ├── time_series.py
│   ├── unsupervised.py
│   ├── model_tuning.py
│   └── explainability.py
│
├── utils/
│   └── visualizations.py
│
└── pages/
    ├── 1_Data_Analysis.py
    ├── 2_Model_Training.py
    └── 3_Predictions.py
```

## 🚀 Installation and Setup

### Prerequisites

- Python 3.10 or later.
- pip package manager.
- Git.

### Step 1: Clone the repository

```bash
git clone https://github.com/lalitha1830/sales_forcasting.git
```

### Step 2: Navigate to the project directory

```bash
cd sales_forcasting
```

### Step 3: Install dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Run the application

```bash
streamlit run app.py
```

The application will open in your browser.

## 🔍 Application Workflow

1. Upload a structured dataset.
2. Explore dataset statistics and visualizations.
3. Select a machine learning task.
4. Choose algorithms or compare available models.
5. Train and evaluate models.
6. Generate predictions using new data.
7. Download prediction results.

## 🎯 Project Objective

The primary objective of ML Studio is to simplify machine learning workflows by combining data analysis, preprocessing, model training, evaluation, explainability and prediction into one accessible platform.

## 🔮 Future Enhancements

- SHAP-based model explanations.
- Advanced automated feature engineering.
- More machine learning algorithms.
- Automated forecasting model selection.
- Model export and deployment.
- Experiment history and model management.
- AI-assisted dataset recommendations.

## 👩‍💻 Author

**Lalitha Tolety**

MSc Computer Science

GitHub: [lalitha1830](https://github.com/lalitha1830)

---

*ML Studio — Explore Data. Train Models. Discover Insights.*
