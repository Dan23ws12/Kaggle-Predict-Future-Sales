import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance
from .data_preprocessor import SalesDataPreprocessor
from . import RAND_STATE, TARGET_COL

data_preprocessor = SalesDataPreprocessor()

def preprocess_and_split_data(data: pd.DataFrame, should_one_hot_encode: bool = True) -> tuple[pd.DataFrame]:
    """
    Preprocesses the sales training data by replacing infrequent values and 
    splitting the data into training and test subsets.
    Returns a tuple containing the training and test subsets data
    """
    data.drop(columns=["item_price_mean"], inplace=True)
    data = data_preprocessor.replace_infrequent_values(data)
    data = data_preprocessor.add_features(data)
    keys = ["shop_id", "item_id", "month_block_num"]
    #aggregate to drop daily item_price and item_cnt_day rows and
    # to reduce data size
    data_grouped = data.groupby(keys, as_index=False).agg({
        TARGET_COL: "first",
        "month_block_length": "first",
        "item_name_length": "first",
        "item_months_sold": "first",
        "avg_item_price_per_month": "first",
        "item_price_median": "median",
    })
    return data_preprocessor.split_data(data_grouped, should_one_hot_encode)

def split_preprocessed_data(data: pd.DataFrame, should_one_hot_encode: bool = True) -> tuple[pd.DataFrame]:
    """
    Splits the preprocessed data into training and test subsets.
    Returns a tuple containing the training and test subsets data
    """
    return data_preprocessor.split_data(data, should_one_hot_encode)

def get_preprocessed_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocesses the sales training data by replacing infrequent values and 
    adding features.
    Returns the preprocessed data.
    """
    data.drop(columns=["item_price_mean"], inplace=True)
    new_data = data_preprocessor.replace_infrequent_values(data)
    new_data = data_preprocessor.add_features(new_data)
    keys = ["shop_id", "item_id", "month_block_num"]
    #aggregate to drop daily item_price and item_cnt_day rows and
    # to reduce data size
    data_grouped = new_data.groupby(keys, as_index=False).agg({
        TARGET_COL: "first",
        "month_block_length": "first",
        "item_name_length": "first",
        "item_months_sold": "first",
        "avg_item_price_per_month": "first",
        "item_price_median": "median",
    })
    return data_grouped

def _model_feature_names(model, n_features: int) -> list[str]:
    """Return feature names stored on a fitted model, or generic fallback names."""
    if getattr(model, "feature_names_in_", None) is not None:
        return list(model.feature_names_in_)
    if getattr(model, "feature_name_", None) is not None:
        return list(model.feature_name_)
    if hasattr(model, "feature_name") and callable(model.feature_name):
        try:
            return list(model.feature_name())
        except Exception:
            pass
    if hasattr(model, "get_booster"):
        booster_names = getattr(model.get_booster(), "feature_names", None)
        if booster_names is not None:
            return list(booster_names)
    return [f"feature_{i}" for i in range(n_features)]


def _hist_gradient_boosting_importances(model) -> np.ndarray:
    """Sum split gain per feature from a fitted HistGradientBoosting model."""
    n_features = model.n_features_in_
    importances = np.zeros(n_features, dtype=np.float64)
    for predictors_at_iteration in model._predictors:
        for predictor in predictors_at_iteration:
            nodes = predictor.nodes
            internal = nodes[nodes["is_leaf"] == 0]
            for feature_idx, gain in zip(internal["feature_idx"], internal["gain"]):
                if 0 <= feature_idx < n_features:
                    importances[feature_idx] += gain
    total = importances.sum()
    if total > 0:
        importances = (100 * importances) / total
    return importances


def _model_feature_importances(model) -> np.ndarray:
    is_lightgbm = type(model).__module__.startswith("lightgbm")
    if hasattr(model, "feature_importances_"):
        importances = np.asarray(model.feature_importances_, dtype=np.float64)
    elif hasattr(model, "feature_importance") and callable(model.feature_importance):
        importances = np.asarray(model.feature_importance(), dtype=np.float64) * 100
        is_lightgbm = True
    elif hasattr(model, "_predictors"):
        return _hist_gradient_boosting_importances(model)
    else:
        raise TypeError(
            "Expected a fitted RandomForest, HistGradientBoosting, LightGBM, "
            "or XGBoost regression model."
        )
    if is_lightgbm:
        total = importances.sum()
        if total > 0:
            importances = 100.0 * importances / total
    return importances


def get_feature_importance(model) -> pd.DataFrame:
    """
    Returns a DataFrame of feature names and importances for a fitted
    HistGradientBoosting, RandomForest, LightGBM, or XGBoost regressor,
    sorted from highest importance to lowest.
    """
    importances = _model_feature_importances(model)
    names = _model_feature_names(model, n_features=len(importances))
    if len(names) != len(importances):
        names = [f"feature_{i}" for i in range(len(importances))]
    return (
        pd.DataFrame({"feature": names, "importance": importances})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


def get_permutation_importance(
    model,
    X,
    y,
    n_repeats: int = 10,
    scoring: str = None,
):
    """
    Plots permutation importance for a fitted HistGradientBoosting,
    RandomForest, LightGBM, or XGBoost regressor.

    X and y should be the evaluation features and target (typically the
    test set). Each bar is the mean drop in score when that feature is
    shuffled; whiskers show the standard deviation across repeats.
    """
    result = permutation_importance(
        model,
        X,
        y,
        n_repeats=n_repeats,
        random_state=RAND_STATE,
        scoring=scoring,
        n_jobs=-2,
    )
    if hasattr(X, "columns"):
        names = list(X.columns)
    else:
        names = _model_feature_names(model, n_features=len(result.importances_mean))
        if len(names) != len(result.importances_mean):
            names = [f"feature_{i}" for i in range(len(result.importances_mean))]

    order = np.argsort(result.importances_mean)
    ordered_names = np.asarray(names)[order]
    fig, ax = plt.subplots(figsize=(8, max(3.5, 0.35 * len(names) + 1)))
    ax.barh(
        ordered_names,
        result.importances_mean[order],
        xerr=result.importances_std[order],
        capsize=3,
    )
    ax.set_xlabel("Permutation importance")
    ax.set_ylabel("Feature")
    ax.set_title("Permutation feature importance")
    fig.tight_layout()
    plt.show()
    return fig, ax

