
import pandas as pd
import plotly.express as px


def plot_dataset_distribution(df, column):
    if column not in df.columns:
        raise ValueError("Column not found.")

    if pd.api.types.is_numeric_dtype(df[column]):
        fig = px.histogram(
            df,
            x=column,
            title=f"Distribution of {column}",
            marginal="box"
        )
    else:
        counts = (
            df[column]
            .astype(str)
            .value_counts()
            .head(20)
            .reset_index()
        )
        counts.columns = [column, "Count"]

        fig = px.bar(
            counts,
            x=column,
            y="Count",
            title=f"Distribution of {column}"
        )

    return fig


def plot_missing_values(df):
    missing = (
        df.isnull()
        .sum()
        .reset_index()
    )

    missing.columns = ["Column", "Missing Values"]
    missing = missing[missing["Missing Values"] > 0]

    if missing.empty:
        return None

    fig = px.bar(
        missing,
        x="Column",
        y="Missing Values",
        title="Missing Values by Column"
    )

    return fig


def plot_correlation_heatmap(df):
    numeric_data = df.select_dtypes(include="number")

    if numeric_data.shape[1] < 2:
        return None

    correlation = numeric_data.corr()

    fig = px.imshow(
        correlation,
        text_auto=".2f",
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
        title="Feature Correlation Heatmap"
    )

    return fig


def plot_model_comparison(comparison_df, metric):
    if metric not in comparison_df.columns:
        raise ValueError("Selected metric not found.")

    fig = px.bar(
        comparison_df,
        x="Model",
        y=metric,
        color="Model",
        title=f"Model Comparison - {metric}",
        text_auto=".3f"
    )

    fig.update_layout(
        showlegend=False,
        xaxis_tickangle=-30
    )

    return fig


def plot_feature_importance(importance_df):
    if importance_df.empty:
        return None

    fig = px.bar(
        importance_df.sort_values("Importance"),
        x="Importance",
        y="Feature",
        orientation="h",
        title="Feature Importance",
        error_x="Importance_Std"
        if "Importance_Std" in importance_df.columns
        else None
    )

    return fig


def plot_actual_vs_predicted(actual, predicted):
    comparison = pd.DataFrame({
        "Actual": list(actual),
        "Predicted": list(predicted)
    })

    fig = px.scatter(
        comparison,
        x="Actual",
        y="Predicted",
        title="Actual vs Predicted Values"
    )

    minimum = min(
        comparison["Actual"].min(),
        comparison["Predicted"].min()
    )
    maximum = max(
        comparison["Actual"].max(),
        comparison["Predicted"].max()
    )

    fig.add_shape(
        type="line",
        x0=minimum,
        y0=minimum,
        x1=maximum,
        y1=maximum,
        line=dict(dash="dash")
    )

    return fig


def plot_forecast(actual, forecast):
    fig = px.line(
        title="Time Series Forecast"
    )

    fig.add_scatter(
        x=actual.index,
        y=actual.values,
        mode="lines",
        name="Historical Data"
    )

    fig.add_scatter(
        x=forecast.index,
        y=forecast.values,
        mode="lines",
        name="Forecast"
    )

    return fig
