"""Tests for rare shop_id and item_id replacement in SalesDataPreprocessor.

These tests check that the preprocessor keeps only the most common IDs
(those whose counts add up to at least 90% of the rows) and labels the
rest as "other".
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

SRC_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SRC_DIR))

from helpers import CAT_FEATURES
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


class TestGetTopValsByCol:
    """Tests for get_top_vals_by_col: which IDs count as the top 90%."""

    def test_returns_highest_frequency_values_until_90_percent_coverage(
        self, preprocessor
    ):
        """Keep the most common IDs until their counts cover 90% of rows.

        With counts 50, 30, 15, and 5, the first three IDs cover 95 rows
        and the rarest ID (count 5) must be left out.
        """
        shop_ids = [1] * 50 + [2] * 30 + [3] * 15 + [4] * 5
        df = pd.DataFrame({"shop_id": shop_ids})

        result = preprocessor.get_top_vals_by_col(df, "shop_id", increment=1)
        expected = top_values_covering_90_percent(df["shop_id"])

        pd.testing.assert_frame_equal(
            result.reset_index(drop=True), expected, check_dtype=False
        )
        assert list(result.columns) == ["shop_id", "frequency"]
        assert result["frequency"].is_monotonic_decreasing
        assert result["frequency"].sum() >= 0.9 * len(df)
        assert 4 not in set(result["shop_id"])

    def test_includes_value_that_crosses_the_90_percent_threshold(self, preprocessor):
        """Keep the ID that first pushes the running count to 90% of rows.

        Nine copies of ID 1 already make up 90% of 10 rows, so ID 2 is
        not included.
        """
        df = pd.DataFrame({"item_id": [1] * 9 + [2]})

        result = preprocessor.get_top_vals_by_col(df, "item_id", increment=1)

        assert list(result["item_id"]) == [1]
        assert list(result["frequency"]) == [9]
        assert result["frequency"].sum() >= 0.9 * len(df)

    def test_stops_once_cumulative_frequency_reaches_90_percent_of_rows(
        self, preprocessor
    ):
        """Do not keep extra IDs after the 90% coverage target is met.

        Ten IDs that each appear once need only nine of them to cover 90%
        of the rows, so the last ID is excluded.
        """
        df = pd.DataFrame({"shop_id": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]})

        result = preprocessor.get_top_vals_by_col(df, "shop_id", increment=1)
        expected = top_values_covering_90_percent(df["shop_id"])

        pd.testing.assert_frame_equal(
            result.reset_index(drop=True), expected, check_dtype=False
        )
        assert len(result) == 9
        assert result["frequency"].sum() >= 0.9 * len(df)

    def test_increment_does_not_include_values_beyond_the_90_percent_cutoff(
        self, preprocessor
    ):
        """A larger lookup step still returns the same 90% cutoff, not extra IDs.

        The function can scan the frequency table in chunks (here, size 2)
        for speed, but the returned IDs must still be exactly those needed
        to reach 90% coverage.
        """
        shop_ids = [1] * 50 + [2] * 30 + [3] * 15 + [4] * 5
        df = pd.DataFrame({"shop_id": shop_ids})
        expected = top_values_covering_90_percent(df["shop_id"])

        result = preprocessor.get_top_vals_by_col(df, "shop_id", increment=2)

        pd.testing.assert_frame_equal(
            result.reset_index(drop=True), expected, check_dtype=False
        )


class TestReplaceValsByCol:
    """Tests for replace_vals_by_col: swapping rare IDs for the label 'other'."""

    def test_replaces_only_values_missing_from_series_with_other(self, preprocessor):
        """Replace IDs that are not in the keep-list with 'other'.

        IDs that are in the Series stay as they are (as strings). Columns
        other than the one being replaced must not change.
        """
        df = pd.DataFrame(
            {
                "shop_id": [10, 20, 30, 20],
                "item_id": [100, 200, 300, 200],
                "item_price": [1.5, 2.5, 3.5, 4.5],
            }
        )
        original_other_cols = df[["item_id", "item_price"]].copy()
        top_vals = pd.Series([10, 20], name="shop_id")

        result = preprocessor.replace_vals_by_col(df.copy(), "shop_id", top_vals)

        assert list(result["shop_id"]) == ["10", "20", "other", "20"]
        pd.testing.assert_frame_equal(
            result[["item_id", "item_price"]].reset_index(drop=True),
            original_other_cols.reset_index(drop=True),
        )

    def test_colname_is_string_dtype_including_kept_values(self, preprocessor):
        """The replaced column is all strings, including IDs that were kept.

        Numeric IDs such as 1 become '1', so the column can mix kept IDs
        with the label 'other'.
        """
        df = pd.DataFrame({"shop_id": [1, 2, 3], "item_id": [7, 8, 9]})

        result = preprocessor.replace_vals_by_col(
            df.copy(), "shop_id", pd.Series([1, 2])
        )

        assert pd.api.types.is_string_dtype(result["shop_id"]) or result[
            "shop_id"
        ].map(lambda value: isinstance(value, str)).all()
        assert result["shop_id"].tolist() == ["1", "2", "other"]
        assert result["item_id"].tolist() == [7, 8, 9]

    def test_raises_when_column_is_missing(self, preprocessor):
        """Raise ValueError if the named column is not in the table."""
        df = pd.DataFrame({"item_id": [1, 2]})

        with pytest.raises(ValueError, match="shop_id is not a column"):
            preprocessor.replace_vals_by_col(df, "shop_id", pd.Series([1]))


class TestReplaceInfrequentValues:
    """Tests for replace_infrequent_values on shop_id and item_id."""

    def test_sample_from_clean_csv_has_expected_format(self, csv_sample):
        """The CSV sample uses the same columns as sales_train_cnt_abs.csv."""
        assert_same_csv_format(csv_sample)
        assert len(csv_sample) == 2000

    def test_uninitialized_preprocessor_replaces_infrequent_categorical_values(
        self,
    ):
        """Work when initialize() has not been called yet.

        The method should compute the top 90% shop_id and item_id values
        from the given table, mark rarer IDs as 'other', and leave other
        columns unchanged.
        """
        preprocessor = SalesDataPreprocessor()
        assert preprocessor.isInitialized is False

        df = make_sales_df(
            shop_ids=[10] * 50 + [20] * 30 + [30] * 15 + [40] * 5,
            item_ids=[100] * 50 + [200] * 30 + [300] * 15 + [400] * 5,
        )
        assert_same_csv_format(df)

        result = preprocessor.replace_infrequent_values(df.copy())

        assert preprocessor.isInitialized is True
        expected_shops = set(top_values_covering_90_percent(df["shop_id"])["shop_id"])
        expected_items = set(top_values_covering_90_percent(df["item_id"])["item_id"])
        result_shops = set(result["shop_id"]) - {"other"}
        result_items = set(result["item_id"]) - {"other"}
        assert {int(value) for value in result_shops} == expected_shops
        assert {int(value) for value in result_items} == expected_items
        assert "other" in set(result["shop_id"])
        assert "other" in set(result["item_id"])
        pd.testing.assert_series_equal(
            result["item_price"].reset_index(drop=True),
            df["item_price"].reset_index(drop=True),
        )
        pd.testing.assert_series_equal(
            result["month_block_num"].reset_index(drop=True),
            df["month_block_num"].reset_index(drop=True),
        )
        pd.testing.assert_series_equal(
            result["item_cnt_month"].reset_index(drop=True),
            df["item_cnt_month"].reset_index(drop=True),
        )

    def test_initialized_preprocessor_replaces_infrequent_categorical_values(
        self,
    ):
        """Work when initialize() has already been called on the same table.

        Rare shop_id and item_id values should still become 'other' using
        the top-value lists stored during initialize().
        """
        preprocessor = SalesDataPreprocessor()
        df = make_sales_df(
            shop_ids=[10] * 50 + [20] * 30 + [30] * 15 + [40] * 5,
            item_ids=[100] * 50 + [200] * 30 + [300] * 15 + [400] * 5,
        )
        assert_same_csv_format(df)

        preprocessor.initialize(df)
        assert preprocessor.isInitialized is True

        result = preprocessor.replace_infrequent_values(df.copy())

        expected_shops = set(top_values_covering_90_percent(df["shop_id"])["shop_id"])
        expected_items = set(top_values_covering_90_percent(df["item_id"])["item_id"])
        result_shops = {int(value) for value in set(result["shop_id"]) - {"other"}}
        result_items = {int(value) for value in set(result["item_id"]) - {"other"}}
        assert result_shops == expected_shops
        assert result_items == expected_items
        assert "other" in set(result["shop_id"])
        assert "other" in set(result["item_id"])

    def test_initialized_and_uninitialized_paths_agree_on_the_same_frame(self):
        """Initialized and uninitialized calls return the same table.

        If both paths see the same sales data, the replaced shop_id and
        item_id columns should match.
        """
        df = make_sales_df(
            shop_ids=[10] * 50 + [20] * 30 + [30] * 15 + [40] * 5,
            item_ids=[100] * 50 + [200] * 30 + [300] * 15 + [400] * 5,
        )

        uninitialized = SalesDataPreprocessor()
        initialized = SalesDataPreprocessor()
        initialized.initialize(df)

        result_uninitialized = uninitialized.replace_infrequent_values(df.copy())
        result_initialized = initialized.replace_infrequent_values(df.copy())

        pd.testing.assert_frame_equal(result_uninitialized, result_initialized)

    def test_initialized_preprocessor_uses_stored_top_values_not_current_frequencies(
        self,
    ):
        """After initialize(), keep using that table's top IDs, not later counts.

        IDs that were rare when initialize() ran stay 'other' even if they
        are common in a later table.
        """
        preprocessor = SalesDataPreprocessor()
        train_df = make_sales_df(
            shop_ids=[10] * 50 + [20] * 30 + [30] * 15 + [40] * 5,
            item_ids=[100] * 91 + [200] * 9,
        )
        later_df = make_sales_df(
            shop_ids=[40] * 80 + [10] * 20,
            item_ids=[200] * 80 + [100] * 20,
        )
        assert_same_csv_format(later_df)

        preprocessor.initialize(train_df)
        result = preprocessor.replace_infrequent_values(later_df.copy())

        stored_shops = set(preprocessor.top_vals_by_col["shop_id"]["shop_id"])
        stored_items = set(preprocessor.top_vals_by_col["item_id"]["item_id"])
        assert 40 not in stored_shops
        assert list(result["shop_id"]).count("other") == 80
        assert {int(value) for value in set(result["shop_id"]) - {"other"}} <= stored_shops
        assert {int(value) for value in set(result["item_id"]) - {"other"}} <= stored_items

    def test_works_on_a_sample_of_sales_train_cnt_abs(self, csv_sample):
        """Run the full replacement on a real sample of sales_train_cnt_abs.csv.

        Rare shop_id and item_id values become 'other'; date, month,
        price, and sales columns stay the same.
        """
        preprocessor = SalesDataPreprocessor()
        assert preprocessor.isInitialized is False
        assert_same_csv_format(csv_sample)

        result = preprocessor.replace_infrequent_values(csv_sample.copy())

        assert preprocessor.isInitialized is True
        for col in CAT_FEATURES:
            expected_ids = set(top_values_covering_90_percent(csv_sample[col])[col])
            kept_ids = {int(value) for value in set(result[col]) - {"other"}}
            assert kept_ids == expected_ids
            assert pd.api.types.is_string_dtype(result[col]) or result[col].map(
                lambda value: isinstance(value, str)
            ).all()
        for col in ["date", "month_block_num", "item_price", "item_cnt_month"]:
            pd.testing.assert_series_equal(
                result[col].reset_index(drop=True),
                csv_sample[col].reset_index(drop=True),
            )
        assert list(result.columns) == EXPECTED_COLUMNS
