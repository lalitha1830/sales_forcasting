
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

from core.preprocessing import (
    split_features_target,
    build_preprocessor
)

from core.model_registry import get_model_registry


def train_automl(
    df,
    target_column,
    task,
    selected_models=None,
    test_size=0.2,
    random_state=42
):
    """
    Train and compare multiple regression or
    classification algorithms automatically.

    Returns trained models, comparison results,
    the best model and any training failures.
    """

    if task not in ["Regression", "Classification"]:
        raise ValueError(
            "AutoML currently supports Regression "
            "and Classification."
        )

    if target_column not in df.columns:
        raise ValueError("Target column not found.")

    # Remove infinite values and rows with missing targets.
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=[target_column]).copy()

    if len(df) < 10:
        raise ValueError(
            "At least 10 valid rows are required."
        )

    # Prepare features and target.
    X, y = split_features_target(
        df,
        target_column
    )

    if X.shape[1] == 0:
        raise ValueError("No usable input features found.")

    if task == "Regression":
        y = pd.to_numeric(y, errors="coerce")
        valid = y.notna()

        X = X.loc[valid]
        y = y.loc[valid]

        if y.nunique() < 2:
            raise ValueError(
                "Regression requires at least two different "
                "target values."
            )

    else:
        if y.nunique() < 2:
            raise ValueError(
                "Classification requires at least two classes."
            )

    # Split data chronologically only in the separate
    # forecasting workflow. Here we use a normal split.
    stratify = None

    if task == "Classification":
        class_counts = y.value_counts()
        number_of_classes = len(class_counts)
        test_count = int(np.ceil(len(y) * test_size))
        train_count = len(y) - test_count

        if (
            class_counts.min() >= 2
            and test_count >= number_of_classes
            and train_count >= number_of_classes
        ):
            stratify = y

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify
    )

    registry = get_model_registry()

    if task not in registry:
        raise ValueError("No algorithms available for this task.")

    available_models = registry[task]

    if selected_models:
        models_to_train = {
            name: factory
            for name, factory in available_models.items()
            if name in selected_models
        }
    else:
        models_to_train = available_models

    if not models_to_train:
        raise ValueError("No valid algorithms were selected.")

    results = []
    trained_models = {}
    failures = []

    for model_name, model_factory in models_to_train.items():

        try:
            # Each model gets its own preprocessing pipeline.
            preprocessor = build_preprocessor(
                X_train,
                scale=True
            )

            model = model_factory()

            pipeline = Pipeline([
                ("preprocessing", preprocessor),
                ("model", model)
            ])

            pipeline.fit(X_train, y_train)

            predictions = pipeline.predict(X_test)

            if task == "Regression":

                mae = mean_absolute_error(
                    y_test,
                    predictions
                )

                mse = mean_squared_error(
                    y_test,
                    predictions
                )

                rmse = float(np.sqrt(mse))

                r2 = r2_score(
                    y_test,
                    predictions
                )

                metrics = {
                    "Model": model_name,
                    "MAE": float(mae),
                    "MSE": float(mse),
                    "RMSE": rmse,
                    "R2": float(r2)
                }

                score = rmse

            else:

                accuracy = accuracy_score(
                    y_test,
                    predictions
                )

                precision = precision_score(
                    y_test,
                    predictions,
                    average="weighted",
                    zero_division=0
                )

                recall = recall_score(
                    y_test,
                    predictions,
                    average="weighted",
                    zero_division=0
                )

                f1 = f1_score(
                    y_test,
                    predictions,
                    average="weighted",
                    zero_division=0
                )

                metrics = {
                    "Model": model_name,
                    "Accuracy": float(accuracy),
                    "Precision": float(precision),
                    "Recall": float(recall),
                    "F1 Score": float(f1)
                }

                score = f1

            results.append(metrics)

            trained_models[model_name] = pipeline

        except Exception as error:

            failures.append({
                "Model": model_name,
                "Error": str(error)
            })

    if not results:
        raise RuntimeError(
            "All selected models failed. "
            "Check your dataset and selected algorithms. "
            f"Errors: {failures}"
        )

    results_df = pd.DataFrame(results)

    # Select the best model using the appropriate metric.
    if task == "Regression":
        results_df = results_df.sort_values(
            by="RMSE",
            ascending=True
        )
    else:
        results_df = results_df.sort_values(
            by="F1 Score",
            ascending=False
        )

    best_model_name = results_df.iloc[0]["Model"]
    best_model = trained_models[best_model_name]

    return {
        "best_model": best_model,
        "best_model_name": best_model_name,
        "trained_models": trained_models,
        "comparison": results_df.reset_index(drop=True),
        "failures": failures,
        "task": task,
        "target_column": target_column,
        "training_rows": len(X_train),
        "testing_rows": len(X_test)
    }
