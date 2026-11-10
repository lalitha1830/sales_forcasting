
import numpy as np
import pandas as pd

from sklearn.cluster import (
    KMeans,
    DBSCAN,
    AgglomerativeClustering
)
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score

from core.preprocessing import build_preprocessor


def prepare_unsupervised_data(df, exclude_columns=None):
    if df is None or df.empty:
        raise ValueError("Please provide a non-empty dataset.")

    data = df.copy()

    if exclude_columns:
        data = data.drop(
            columns=exclude_columns,
            errors="ignore"
        )

    data = data.replace([np.inf, -np.inf], np.nan)

    if data.shape[1] == 0:
        raise ValueError("No usable features were found.")

    preprocessor = build_preprocessor(data)

    try:
        transformed = preprocessor.fit_transform(data)
    except Exception as error:
        raise ValueError(
            f"Data preprocessing failed: {error}"
        )

    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()

    return transformed, preprocessor


def perform_clustering(
    X,
    algorithm="KMeans",
    n_clusters=3
):
    if algorithm == "KMeans":
        model = KMeans(
            n_clusters=n_clusters,
            random_state=42,
            n_init=10
        )

    elif algorithm == "DBSCAN":
        model = DBSCAN(
            eps=0.5,
            min_samples=5
        )

    elif algorithm == "Agglomerative":
        model = AgglomerativeClustering(
            n_clusters=n_clusters
        )

    else:
        raise ValueError("Unsupported clustering algorithm.")

    labels = model.fit_predict(X)

    unique_labels = set(labels)
    valid_labels = unique_labels - {-1}

    score = None

    if (
        len(valid_labels) >= 2
        and len(unique_labels) > 1
        and len(X) <= 10000
    ):
        try:
            score = silhouette_score(X, labels)
        except Exception:
            score = None

    return {
        "model": model,
        "labels": labels,
        "silhouette_score": score,
        "number_of_clusters": len(valid_labels),
        "noise_points": int(np.sum(labels == -1))
    }


def detect_anomalies(X, contamination=0.05):
    model = IsolationForest(
        contamination=contamination,
        random_state=42
    )

    predictions = model.fit_predict(X)

    # 1 = Normal, -1 = Anomaly
    labels = np.where(predictions == -1, 1, 0)

    return {
        "model": model,
        "labels": labels,
        "normal_records": int(np.sum(labels == 0)),
        "anomalies": int(np.sum(labels == 1))
    }


def reduce_dimensions(
    X,
    method="PCA",
    n_components=2
):
    if n_components < 1 or n_components > min(X.shape):
        raise ValueError(
            "Invalid number of components for this dataset."
        )

    if method == "PCA":
        model = PCA(
            n_components=n_components,
            random_state=42
        )
        reduced = model.fit_transform(X)

        explained_variance = float(
            np.sum(model.explained_variance_ratio_)
        )

    elif method == "t-SNE":
        if len(X) < 4:
            raise ValueError(
                "t-SNE requires at least four records."
            )

        model = TSNE(
            n_components=n_components,
            random_state=42,
            init="pca",
            learning_rate="auto"
        )
        reduced = model.fit_transform(X)
        explained_variance = None

    else:
        raise ValueError(
            "Supported methods are PCA and t-SNE."
        )

    columns = [
        f"Component_{i + 1}"
        for i in range(n_components)
    ]

    result = pd.DataFrame(
        reduced,
        columns=columns
    )

    return {
        "model": model,
        "reduced_data": result,
        "explained_variance": explained_variance
    }
