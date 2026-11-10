
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.model_selection import (
    GridSearchCV,
    KFold,
    StratifiedKFold
)

from core.model_registry import create_model


PARAMETER_GRIDS = {
    "Linear Regression": {},

    "Ridge Regression": {
        "model__alpha": [0.1, 1.0, 10.0]
    },

    "Lasso Regression": {
        "model__alpha": [0.01, 0.1, 1.0]
    },

    "Decision Tree": {
        "model__max_depth": [None, 5, 10, 20],
        "model__min_samples_split": [2, 5, 10]
    },

    "Random Forest": {
        "model__n_estimators": [100, 200],
        "model__max_depth": [None, 10, 20]
    },

    "Extra Trees": {
        "model__n_estimators": [100, 200],
        "model__max_depth": [None, 10, 20]
    },

    "Gradient Boosting": {
        "model__n_estimators": [100, 200],
        "model__learning_rate": [0.01, 0.1]
    },

    "Logistic Regression": {
        "model__C": [0.1, 1.0, 10.0]
    },

    "Support Vector Machine": {
        "model__C": [0.1, 1.0, 10.0]
    },

    "K-Nearest Neighbors": {
        "model__n_neighbors": [3, 5, 7]
    }
}


def tune_model(
    X,
    y,
    task,
    model_name,
    preprocessor,
    cv=3
):
    if task not in ["Regression", "Classification"]:
        raise ValueError(
            "Tuning currently supports Regression and Classification."
        )

    if model_name not in PARAMETER_GRIDS:
        raise ValueError(
            f"No tuning configuration available for {model_name}."
        )

    parameter_grid = PARAMETER_GRIDS[model_name]

    if not parameter_grid:
        raise ValueError(
            f"{model_name} has no tuning parameters configured."
        )

    if task == "Classification":
        minimum_class_count = y.value_counts().min()

        if minimum_class_count < 2:
            raise ValueError(
                "Each class needs at least two records for cross-validation."
            )

        folds = min(cv, int(minimum_class_count))

        cross_validator = StratifiedKFold(
            n_splits=folds,
            shuffle=True,
            random_state=42
        )

        scoring = "f1_weighted"

    else:
        folds = min(cv, len(X))

        if folds < 2:
            raise ValueError(
                "At least two records are required for tuning."
            )

        cross_validator = KFold(
            n_splits=folds,
            shuffle=True,
            random_state=42
        )

        scoring = "neg_root_mean_squared_error"

    model = create_model(task, model_name)

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    search = GridSearchCV(
        estimator=pipeline,
        param_grid=parameter_grid,
        scoring=scoring,
        cv=cross_validator,
        n_jobs=-1,
        error_score="raise"
    )

    search.fit(X, y)

    results = pd.DataFrame(search.cv_results_)

    return {
        "best_model": search.best_estimator_,
        "best_parameters": search.best_params_,
        "best_score": float(search.best_score_),
        "results": results.sort_values(
            "rank_test_score"
        )
    }
