"""Shared fixtures and helpers for SalesDataPreprocessor tests."""

import sys
from pathlib import Path

import pandas as pd
import pytest

SRC_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SRC_DIR))

from helpers.data_preprocessor import SalesDataPreprocessor

SALES_TRAIN_CNT_ABS = REPO_ROOT / "data" / "clean data" / "sales_train_cnt_abs.csv"
EXPECTED_COLUMNS = [
    "date",
    "month_block_num",
    "shop_id",
    "item_id",
    "item_price",
    "item_cnt_month",
]


def top_values_covering_90_percent(series: pd.Series) -> pd.DataFrame:
    """Build the expected top-ID table used to check get_top_vals_by_col.

    Counts how often each value appears, sorts those counts from high to
    low, and keeps values only until their counts add up to at least 90%
    of the rows. Everything after that cutoff should be treated as rare.
    """
    frequency_df = series.value_counts().reset_index()
    frequency_df.columns = [series.name, "frequency"]
    frequency_df = frequency_df.sort_values(
        by="frequency", ascending=False
    ).reset_index(drop=True)
    target = 0.9 * len(series)
    cumsum = 0
    for i, freq in enumerate(frequency_df["frequency"]):
        cumsum += freq
        if cumsum >= target:
            return frequency_df.iloc[: i + 1].reset_index(drop=True)
    return frequency_df.reset_index(drop=True)


def make_sales_df(shop_ids, item_ids) -> pd.DataFrame:
    """Build a small sales table with the same columns as sales_train_cnt_abs.csv.

    Used so replace_infrequent_values can be tested on a known mix of
    common and rare shop_id / item_id values without loading the full CSV.
    """
    n = len(shop_ids)
    return pd.DataFrame(
        {
            "date": pd.to_datetime(["2013-01-02"] * n),
            "month_block_num": [0] * n,
            "shop_id": shop_ids,
            "item_id": item_ids,
            "item_price": [999.0] * n,
            "item_cnt_month": [1.0] * n,
        }
    )


def assert_same_csv_format(df: pd.DataFrame) -> None:
    """Fail if the table does not use the sales_train_cnt_abs.csv column names."""
    assert list(df.columns) == EXPECTED_COLUMNS


@pytest.fixture
def csv_sample() -> pd.DataFrame:
    """First 2,000 rows of sales_train_cnt_abs.csv, used as a realistic sample."""
    sample = pd.read_csv(SALES_TRAIN_CNT_ABS, nrows=2000)
    assert_same_csv_format(sample)
    return sample


@pytest.fixture
def preprocessor() -> SalesDataPreprocessor:
    """A fresh preprocessor that has not been initialized yet."""
    return SalesDataPreprocessor()


@pytest.fixture
def expected_columns():
    return EXPECTED_COLUMNS


@pytest.fixture
def top_90():
    return top_values_covering_90_percent


@pytest.fixture
def sales_df_factory():
    return make_sales_df


@pytest.fixture
def assert_csv_format():
    return assert_same_csv_format
