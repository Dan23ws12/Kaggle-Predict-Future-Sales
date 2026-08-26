"""Tests for feature-aggregation helpers on SalesDataPreprocessor.

Each test checks that the original sales columns stay the same and that
the new column matches the intended aggregation.
"""

import pandas as pd

ORIGINAL_COLUMNS = [
    "date",
    "month_block_num",
    "shop_id",
    "item_id",
    "item_price",
    "item_cnt_day",
    "item_cnt_month",
]


def make_aggregation_df() -> pd.DataFrame:
    """Build a small sales table with known dates, prices, and daily counts."""
    return pd.DataFrame(
        {
            "date": [
                "2013-01-02",
                "2013-01-10",
                "2013-02-01",
                "2013-02-05",
                "2013-02-05",
            ],
            "month_block_num": [0, 0, 1, 1, 1],
            "shop_id": [1, 1, 1, 2, 2],
            "item_id": [10, 10, 10, 20, 20],
            "item_price": [100.0, 200.0, 150.0, 50.0, 70.0],
            "item_cnt_day": [1.0, 3.0, 2.0, 4.0, 6.0],
            "item_cnt_month": [4.0, 4.0, 2.0, 10.0, 10.0],
        }
    )


def assert_original_columns_unchanged(result: pd.DataFrame, original: pd.DataFrame) -> None:
    """Fail if any column that existed before the aggregation was changed."""
    pd.testing.assert_frame_equal(
        result[ORIGINAL_COLUMNS].reset_index(drop=True),
        original[ORIGINAL_COLUMNS].reset_index(drop=True),
        check_dtype=False,
    )


class TestAddMonthLength:
    """Tests for add_month_length."""

    def test_adds_day_span_between_earliest_and_latest_date_per_month(
        self, preprocessor
    ):
        """month_block_length is the number of days from first to last date in each month.

        Month 0 runs from 2013-01-02 to 2013-01-10 (8 days). Month 1 runs
        from 2013-02-01 to 2013-02-05 (4 days). Every row in a month gets
        that month's length.
        """
        df = make_aggregation_df()
        result = preprocessor.add_month_length(df.copy())

        assert_original_columns_unchanged(result, df)
        assert list(result["month_block_length"]) == [8, 8, 4, 4, 4]

    def test_single_date_in_a_month_has_length_zero(self, preprocessor):
        """A month with only one date has a length of 0 days."""
        df = pd.DataFrame(
            {
                "date": ["2013-03-15", "2013-03-15"],
                "month_block_num": [2, 2],
                "shop_id": [1, 2],
                "item_id": [10, 20],
                "item_price": [1.0, 2.0],
                "item_cnt_day": [1.0, 1.0],
                "item_cnt_month": [1.0, 1.0],
            }
        )

        result = preprocessor.add_month_length(df.copy())

        assert list(result["month_block_length"]) == [0, 0]


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
                "date": ["2013-01-02", "2013-01-03", "2013-01-04"],
                "month_block_num": [0, 0, 0],
                "shop_id": [1, 1, 1],
                "item_id": [10, 20, "other"],
                "item_price": [1.0, 2.0, 3.0],
                "item_cnt_day": [1.0, 1.0, 1.0],
                "item_cnt_month": [1.0, 1.0, 1.0],
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
                "date": ["2013-01-02"],
                "month_block_num": [0],
                "shop_id": [1],
                "item_id": ["10"],
                "item_price": [1.0],
                "item_cnt_day": [1.0],
                "item_cnt_month": [1.0],
            }
        )

        result = preprocessor.add_item_name_length(df.copy())

        assert list(result["item_name_length"]) == [3]


class TestAddItemMonthsSold:
    """Tests for add_item_months_sold."""

    def test_counts_distinct_months_each_item_appears_in(self, preprocessor):
        """item_months_sold is how many different month blocks an item appears in.

        Item 10 is sold in months 0 and 1, so every item-10 row gets 2.
        Item 20 is sold only in month 1, so those rows get 1.
        """
        df = make_aggregation_df()
        result = preprocessor.add_item_months_sold(df.copy())

        assert_original_columns_unchanged(result, df)
        assert list(result["item_months_sold"]) == [2, 2, 2, 1, 1]


class TestAddAvgItemPricePerMonth:
    """Tests for add_avg_item_price_per_month."""

    def test_averages_item_price_for_each_item_in_each_month(self, preprocessor):
        """avg_item_price_per_month is the mean price of that item in that month.

        Item 10 in month 0: (100 + 200) / 2 = 150. Item 10 in month 1: 150.
        Item 20 in month 1: (50 + 70) / 2 = 60.
        """
        df = make_aggregation_df()
        result = preprocessor.add_avg_item_price_per_month(df.copy())

        assert_original_columns_unchanged(result, df)
        assert list(result["avg_item_price_per_month"]) == [150.0, 150.0, 150.0, 60.0, 60.0]


class TestAddAvgSalesPerShop:
    """Tests for add_avg_sales_per_shop."""

    def test_averages_daily_sales_for_each_shop_in_each_month(self, preprocessor):
        """avg_sales_per_shop is the mean item_cnt_day for that shop in that month.

        Shop 1 in month 0: (1 + 3) / 2 = 2. Shop 1 in month 1: 2.
        Shop 2 in month 1: (4 + 6) / 2 = 5.
        """
        df = make_aggregation_df()
        result = preprocessor.add_avg_sales_per_shop(df.copy())

        assert_original_columns_unchanged(result, df)
        assert list(result["avg_sales_per_shop"]) == [2.0, 2.0, 2.0, 5.0, 5.0]


class TestAddAvgSalesPerItem:
    """Tests for add_avg_sales_per_item."""

    def test_averages_daily_sales_for_each_item_in_each_month(self, preprocessor):
        """avg_sales_per_item is the mean item_cnt_day for that item in that month.

        Item 10 in month 0: (1 + 3) / 2 = 2. Item 10 in month 1: 2.
        Item 20 in month 1: (4 + 6) / 2 = 5.
        """
        df = make_aggregation_df()
        result = preprocessor.add_avg_sales_per_item(df.copy())

        assert_original_columns_unchanged(result, df)
        assert list(result["avg_sales_per_item"]) == [2.0, 2.0, 2.0, 5.0, 5.0]
