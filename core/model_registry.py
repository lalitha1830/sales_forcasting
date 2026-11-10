from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures

# Regression
from sklearn.linear_model import (
    LinearRegression,
    Ridge,
    Lasso,
    ElasticNet
)
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    ExtraTreesRegressor,
    HistGradientBoostingRegressor
)
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor

# Classification
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    ExtraTreesClassifier,
    HistGradientBoostingClassifier
)
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

# Clustering
from sklearn.cluster import (
    KMeans,
    DBSCAN,
    AgglomerativeClustering
)
from sklearn.mixture import GaussianMixture

# Anomaly detection
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM

# Dimensionality reduction
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.manifold import TSNE
class DenseTransformer(BaseEstimator, TransformerMixin):
    """Convert sparse matrices into dense arrays when required."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if sparse.issparse(X):
            return X.toarray()
        return X

def get_model_registry():
    """
    Return all available machine learning algorithms.

    Each value is a factory function that creates
    a fresh model instance.
    """

    registry = {

        "Regression": {
            "Linear Regression": lambda: LinearRegression(),

            "Polynomial Regression": lambda: make_pipeline(
                PolynomialFeatures(degree=2),
                LinearRegression()
            ),

            "Ridge Regression": lambda: Ridge(),

            "Lasso Regression": lambda: Lasso(),

            "Elastic Net": lambda: ElasticNet(),

            "Decision Tree": lambda: DecisionTreeRegressor(
                random_state=42
            ),

            "Random Forest": lambda: RandomForestRegressor(
                n_estimators=100,
                random_state=42
            ),

            "Extra Trees": lambda: ExtraTreesRegressor(
                n_estimators=100,
                random_state=42
            ),

            "Gradient Boosting": lambda: GradientBoostingRegressor(
                random_state=42
            ),

           "Hist Gradient Boosting": lambda: make_pipeline(
                DenseTransformer(),
                HistGradientBoostingRegressor(random_state=42)
            ),

            "Support Vector Regression": lambda: SVR(),

            "K Nearest Neighbors": lambda: KNeighborsRegressor()
        },

        "Classification": {
            "Logistic Regression": lambda: LogisticRegression(
                max_iter=1000
            ),

            "Decision Tree": lambda: DecisionTreeClassifier(
                random_state=42
            ),

            "Random Forest": lambda: RandomForestClassifier(
                n_estimators=100,
                random_state=42
            ),

            "Extra Trees": lambda: ExtraTreesClassifier(
                n_estimators=100,
                random_state=42
            ),

            "Gradient Boosting": lambda: GradientBoostingClassifier(
                random_state=42
            ),

           "Hist Gradient Boosting": lambda: make_pipeline(
                DenseTransformer(),
                HistGradientBoostingClassifier(random_state=42)
            ),

            "Support Vector Machine": lambda: SVC(
                probability=True,
                random_state=42
            ),

            "K Nearest Neighbors": lambda: KNeighborsClassifier(),

            "Gaussian Naive Bayes": lambda: make_pipeline(
                DenseTransformer(),
                GaussianNB()
            )
        },

        "Clustering": {
            "K-Means": lambda: KMeans(
                n_clusters=3,
                random_state=42,
                n_init=10
            ),

            "DBSCAN": lambda: DBSCAN(),

            "Agglomerative Clustering": lambda: AgglomerativeClustering(
                n_clusters=3
            ),

            "Gaussian Mixture": lambda: GaussianMixture(
                n_components=3,
                random_state=42
            )
        },

        "Anomaly Detection": {
            "Isolation Forest": lambda: IsolationForest(
                contamination="auto",
                random_state=42
            ),

            "Local Outlier Factor": lambda: LocalOutlierFactor(
                contamination="auto"
            ),

            "One Class SVM": lambda: OneClassSVM()
        },

        "Dimensionality Reduction": {
            "Principal Component Analysis": lambda: PCA(
                n_components=2
            ),

            "Truncated SVD": lambda: TruncatedSVD(
                n_components=2,
                random_state=42
            ),

            "t-SNE": lambda: TSNE(
                n_components=2,
                random_state=42
            )
        }
    }

    # Optional algorithms: register only if installed.

    try:
        from xgboost import (
            XGBRegressor,
            XGBClassifier
        )

        registry["Regression"]["XGBoost"] = lambda: XGBRegressor(
            n_estimators=100,
            random_state=42,
            verbosity=0
        )

        registry["Classification"]["XGBoost"] = lambda: XGBClassifier(
            n_estimators=100,
            random_state=42,
            eval_metric="logloss",
            verbosity=0
        )

    except ImportError:
        pass

    try:
        from lightgbm import (
            LGBMRegressor,
            LGBMClassifier
        )

        registry["Regression"]["LightGBM"] = lambda: LGBMRegressor(
            random_state=42,
            verbosity=-1
        )

        registry["Classification"]["LightGBM"] = lambda: LGBMClassifier(
            random_state=42,
            verbosity=-1
        )

    except ImportError:
        pass

    try:
        from catboost import (
            CatBoostRegressor,
            CatBoostClassifier
        )

        registry["Regression"]["CatBoost"] = lambda: CatBoostRegressor(
            verbose=0,
            random_seed=42
        )

        registry["Classification"]["CatBoost"] = lambda: CatBoostClassifier(
            verbose=0,
            random_seed=42
        )

    except ImportError:
        pass

    return registry


def get_available_models(task):
    """
    Return algorithm names available for a selected task.
    """

    registry = get_model_registry()

    if task not in registry:
        raise ValueError(
            f"Unsupported ML task: {task}"
        )

    return list(registry[task].keys())


def create_model(task, model_name):
    """
    Create a fresh model instance.
    """

    registry = get_model_registry()

    if task not in registry:
        raise ValueError(
            f"Unsupported ML task: {task}"
        )

    if model_name not in registry[task]:
        raise ValueError(
            f"Model '{model_name}' is unavailable."
        )

    return registry[task][model_name]()
