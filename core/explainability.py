
import pandas as pd

from sklearn.inspection import permutation_importance


def explain_model(
    model,
    X_test,
    y_test,
    task="Regression",
    top_n=20,
    n_repeats=5
):
    """
    Calculate feature importance using permutation importance.

    Works with fitted sklearn estimators and pipelines.
    """

    if model is None:
        raise ValueError("Please provide a trained model.")

    if X_test is None or len(X_test) == 0:
        raise ValueError("Test data is empty.")

    if y_test is None or len(y_test) == 0:
        raise ValueError("Test target values are empty.")

    if task == "Regression":
        scoring = "neg_root_mean_squared_error"

    elif task == "Classification":
        scoring = "f1_weighted"

    else:
        raise ValueError(
            "Explainability currently supports Regression "
            "and Classification."
        )

    try:
        result = permutation_importance(
            model,
            X_test,
            y_test,
            scoring=scoring,
            n_repeats=n_repeats,
            random_state=42,
            n_jobs=-1
        )

        importance_df = pd.DataFrame({
            "Feature": X_test.columns,
            "Importance": result.importances_mean,
            "Importance_Std": result.importances_std
        })

        importance_df = importance_df.sort_values(
            by="Importance",
            ascending=False
        ).reset_index(drop=True)

        return importance_df.head(top_n)

    except Exception as error:
        raise ValueError(
            f"Feature importance calculation failed: {error}"
        )


def get_feature_importance_summary(importance_df):
    if importance_df is None or importance_df.empty:
        raise ValueError("No feature importance results available.")

    most_important = importance_df.iloc[0]

    return {
        "most_important_feature": most_important["Feature"],
        "importance_score": float(most_important["Importance"]),
        "total_features_analyzed": len(importance_df)
    }
