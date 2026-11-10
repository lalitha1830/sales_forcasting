
import numpy as np

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    silhouette_score,
    davies_bouldin_score
)


def evaluate_regression(y_true, y_pred):
    """Calculate regression performance metrics."""

    mse = mean_squared_error(y_true, y_pred)

    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "MSE": float(mse),
        "RMSE": float(np.sqrt(mse)),
        "R2": float(r2_score(y_true, y_pred))
    }


def evaluate_classification(y_true, y_pred):
    """Calculate classification metrics and reports."""

    return {
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "Precision": float(
            precision_score(
                y_true, y_pred,
                average="weighted",
                zero_division=0
            )
        ),
        "Recall": float(
            recall_score(
                y_true, y_pred,
                average="weighted",
                zero_division=0
            )
        ),
        "F1 Score": float(
            f1_score(
                y_true, y_pred,
                average="weighted",
                zero_division=0
            )
        ),
        "Confusion Matrix": confusion_matrix(
            y_true, y_pred
        ).tolist(),
        "Classification Report": classification_report(
            y_true, y_pred,
            zero_division=0,
            output_dict=True
        )
    }


def evaluate_clustering(X, labels):
    """Evaluate clustering quality without true labels."""

    labels = np.asarray(labels)
    unique_labels = set(labels)

    # Noise points (-1) are excluded from cluster count.
    cluster_labels = unique_labels - {-1}

    if len(cluster_labels) < 2:
        return {
            "Silhouette Score": None,
            "Davies-Bouldin Index": None,
            "Message": "At least two non-noise clusters are required."
        }

    mask = labels != -1
    X_valid = X[mask]
    labels_valid = labels[mask]

    if len(set(labels_valid)) >= len(labels_valid):
        return {
            "Silhouette Score": None,
            "Davies-Bouldin Index": None,
            "Message": "Not enough samples per cluster."
        }

    return {
        "Silhouette Score": float(
            silhouette_score(X_valid, labels_valid)
        ),
        "Davies-Bouldin Index": float(
            davies_bouldin_score(X_valid, labels_valid)
        ),
        "Number of Clusters": len(cluster_labels),
        "Noise Points": int(np.sum(labels == -1))
    }


def evaluate_anomaly_detection(
    y_true,
    predictions
):
    """
    Evaluate anomaly predictions when actual labels exist.

    Expected actual labels:
    0 = Normal, 1 = Anomaly

    sklearn anomaly estimators return:
    1 = Normal, -1 = Anomaly
    """

    predictions = np.asarray(predictions)

    converted = np.where(predictions == -1, 1, 0)

    return evaluate_classification(
        y_true,
        converted
    )
