"""Tests for feature-aggregation helpers on SalesDataPreprocessor.

Each test checks that the original sales columns stay the same and that
the new column matches the intended aggregation.
"""

import pandas as pd

# Columns match data/clean data/sales_train_cnt_0.csv
ORIGINAL_COLUMNS = [
    "shop_id",
    "item_id",
    "month_block_num",
    "item_cnt_month",
    "item_price_median",
]


def make_aggregation_df() -> pd.DataFrame:
    """Build a small sales table in the sales_train_cnt_0.csv format.

    item_price_median is the median of the conceptual daily item_price
    values in that month block:
    month 0 prices [100, 200] -> 150; month 1 prices [150, 50, 70] -> 150;
    month 2 prices [80, 120] -> 100.
    """
    return pd.DataFrame(
        {
            "shop_id": [1, 1, 2, 1, 2],
            "item_id": [10, 10, 20, 10, 20],
            "month_block_num": [0, 1, 1, 2, 2],
            "item_cnt_month": [4.0, 2.0, 10.0, 8.0, 1.0],
            "item_price_median": [150.0, 150.0, 150.0, 100.0, 100.0],
        }
    )


def assert_original_columns_unchanged(result: pd.DataFrame, original: pd.DataFrame) -> None:
    """Fail if any column that existed before the aggregation was changed."""
    pd.testing.assert_frame_equal(
        result[ORIGINAL_COLUMNS].reset_index(drop=True),
        original[ORIGINAL_COLUMNS].reset_index(drop=True),
        check_dtype=False,
    )


class TestAddItemNameLength:
    """Tests for add_item_name_length."""

    def test_counts_alphanumeric_characters_and_handles_other(self, preprocessor):
        """item_name_length counts letters and digits only; 'other' is always 5.

        Whitespace and punctuation are ignored. Item 10 is 'Hello World!'
        (10 letters). Item 20 is 'ab-c 12' (5 letters/digits). Rows with
        item_id 'other' get length 5 without looking up a name.
        """
        preprocessor._items_df = pd.DataFrame(
            {
                "item_name": ["Hello World!", "ab-c 12"],
                "item_id": [10, 20],
                "item_category_id": [40, 40],
            }
        )
        df = pd.DataFrame(
            {
                "shop_id": [1, 1, 1],
                "item_id": [10, 20, "other"],
                "month_block_num": [0, 0, 0],
                "item_cnt_month": [1.0, 1.0, 1.0],
                "item_price_median": [1.0, 2.0, 3.0],
            }
        )
        original = df.copy()

        result = preprocessor.add_item_name_length(df.copy())

        assert_original_columns_unchanged(result, original)
        assert list(result["item_name_length"]) == [10, 5, 5]

    def test_looks_up_string_numeric_item_ids(self, preprocessor):
        """String item IDs such as '10' still use the matching items.csv name."""
        preprocessor._items_df = pd.DataFrame(
            {
                "item_name": ["Cat"],
                "item_id": [10],
                "item_category_id": [40],
            }
        )
        df = pd.DataFrame(
            {
                "shop_id": [1],
                "item_id": ["10"],
                "month_block_num": [0],
                "item_cnt_month": [1.0],
                "item_price_median": [1.0],
            }
        )

        result = preprocessor.add_item_name_length(df.copy())

        assert list(result["item_name_length"]) == [3]


class TestAddNumMonthsSoldPrior:
    """Tests for add_num_months_sold_prior."""

    def test_counts_prior_months_each_item_was_sold_in(self, preprocessor):
        """num_months_sold_prior is how many earlier months the item was sold in.

        Item 10 is sold in months 0, 1, and 2: month 0 is 0, month 1 is 1,
        month 2 is 2. Item 20 is sold in months 1 and 2: month 1 is 0, month 2 is 1.
        """
        df = make_aggregation_df()
        result = preprocessor.add_num_months_sold_prior(df.copy())

        assert_original_columns_unchanged(result, df)
        assert list(result["num_months_sold_prior"]) == [0, 1, 0, 2, 1]


class TestAddAvgItemPricePerMonth:
    """Tests for add_avg_item_price_per_month."""

    def test_averages_item_price_median_for_each_item_in_each_month(self, preprocessor):
        """avg_item_price_per_month is the mean item_price_median of that item in that month.

        item_price_median is already the month-block median, so every row
        in month 0 or 1 is 150 and every row in month 2 is 100.
        """
        df = make_aggregation_df()
        result = preprocessor.add_avg_item_price_per_month(df.copy())

        assert_original_columns_unchanged(result, df)
        assert list(result["avg_item_price_per_month"]) == [150.0, 150.0, 150.0, 100.0, 100.0]


class TestAddAvgShopPricePerMonth:
    """Tests for add_avg_shop_price_per_month."""

    def test_averages_item_price_median_for_each_shop_in_each_month(self, preprocessor):
        """avg_shop_price_per_month is the mean item_price_median of that shop in that month.

        item_price_median is already the month-block median, so every row
        in month 0 or 1 is 150 and every row in month 2 is 100.
        """
        df = make_aggregation_df()
        result = preprocessor.add_avg_shop_price_per_month(df.copy())

        assert_original_columns_unchanged(result, df)
        assert list(result["avg_shop_price_per_month"]) == [150.0, 150.0, 150.0, 100.0, 100.0]


class TestAddAvgSalesPerShop:
    """Tests for add_avg_sales_per_shop."""

    def test_rolling_average_of_prior_months_shop_sales(self, preprocessor):
        """avg_sales_per_shop is the expanding mean of that shop's prior monthly sales totals.

        Shop 1 month 0 has no history (0). Shop 1 month 1 uses month 0 total 4.
        Shop 1 month 2 averages month 0 (4) and month 1 (2) = 3.
        Shop 2 first appears in month 1, so that month is 0; month 2 uses 10.
        """
        df = make_aggregation_df()
        result = preprocessor.add_avg_sales_per_shop(df.copy())

        assert_original_columns_unchanged(result, df)
        expected = [0.0, 4.0, 0.0, 3.0, 10.0]
        pd.testing.assert_series_equal(
            result["avg_sales_per_shop"].reset_index(drop=True),
            pd.Series(expected, name="avg_sales_per_shop"),
            check_dtype=False,
        )


class TestAddAvgSalesPerItem:
    """Tests for add_avg_sales_per_item."""

    def test_rolling_average_of_prior_months_item_sales(self, preprocessor):
        """avg_sales_per_item is the expanding mean of that item's prior monthly sales totals.

        Item 10 month 0 has no history (0). Item 10 month 1 uses month 0 total 4.
        Item 10 month 2 averages month 0 (4) and month 1 (2) = 3.
        Item 20 first appears in month 1, so that month is 0; month 2 uses 10.
        """
        df = make_aggregation_df()
        result = preprocessor.add_avg_sales_per_item(df.copy())

        assert_original_columns_unchanged(result, df)
        expected = [0.0, 4.0, 0.0, 3.0, 10.0]
        pd.testing.assert_series_equal(
            result["avg_sales_per_item"].reset_index(drop=True),
            pd.Series(expected, name="avg_sales_per_item"),
            check_dtype=False,
        )
