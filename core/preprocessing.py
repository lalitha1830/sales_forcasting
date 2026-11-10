
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder


def extract_datetime_features(df):
    """
    Convert detected date columns into useful numerical features.
    """

    df = df.copy()

    for column in df.columns:
        name = str(column).lower()

        date_keywords = [
            "date", "time", "day", "month",
            "year", "timestamp"
        ]

        if any(word in name for word in date_keywords):
            parsed = pd.to_datetime(
                df[column], errors="coerce"
            )

            if parsed.notna().mean() >= 0.7:
                df[column] = parsed
                df[f"{column}_year"] = parsed.dt.year
                df[f"{column}_month"] = parsed.dt.month
                df[f"{column}_day"] = parsed.dt.day
                df[f"{column}_weekday"] = parsed.dt.dayofweek

                df.drop(columns=[column], inplace=True)

    return df


def split_features_target(df, target_column=None):
    """
    Separate input features and target.
    Use target_column=None for unsupervised learning.
    """

    df = extract_datetime_features(df)

    if target_column is not None:
        if target_column not in df.columns:
            raise ValueError("Selected target column not found.")

        X = df.drop(columns=[target_column])
        y = df[target_column]

        return X, y

    return df, None


def build_preprocessor(X, scale=True):
    """
    Build a preprocessing pipeline for numerical
    and categorical features.
    """

    numeric_columns = X.select_dtypes(
        include=["number"]
    ).columns.tolist()

    categorical_columns = X.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()

    numeric_steps = [
        ("imputer", SimpleImputer(strategy="median"))
    ]

    if scale:
        numeric_steps.append(
            ("scaler", StandardScaler())
        )

    numeric_pipeline = Pipeline(numeric_steps)

    categorical_pipeline = Pipeline([
        (
            "imputer",
            SimpleImputer(
                strategy="constant",
                fill_value="Missing"
            )
        ),
        (
            "encoder",
            OneHotEncoder(handle_unknown="ignore")
        )
    ])

    transformers = []

    if numeric_columns:
        transformers.append(
            ("numeric", numeric_pipeline, numeric_columns)
        )

    if categorical_columns:
        transformers.append(
            ("categorical", categorical_pipeline,
             categorical_columns)
        )

    if not transformers:
        raise ValueError(
            "No usable numerical or categorical features found."
        )

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop"
    )

    return preprocessor
