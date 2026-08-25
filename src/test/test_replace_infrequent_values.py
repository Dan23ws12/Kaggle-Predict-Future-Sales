"""Tests for replace_infrequent_values on shop_id and item_id."""

import pandas as pd

from helpers import CAT_FEATURES
from helpers.data_preprocessor import SalesDataPreprocessor


class TestReplaceInfrequentValues:
    def test_sample_from_clean_csv_has_expected_format(self, csv_sample, assert_csv_format):
        """The CSV sample uses the same columns as sales_train_cnt_abs.csv."""
        assert_csv_format(csv_sample)
        assert len(csv_sample) == 2000

    def test_uninitialized_preprocessor_replaces_infrequent_categorical_values(
        self, sales_df_factory, assert_csv_format, top_90
    ):
        """Work when initialize() has not been called yet.

        The method should compute the top 90% shop_id and item_id values
        from the given table, mark rarer IDs as 'other', and leave other
        columns unchanged.
        """
        preprocessor = SalesDataPreprocessor()
        assert preprocessor.isInitialized is False

        df = sales_df_factory(
            shop_ids=[10] * 50 + [20] * 30 + [30] * 15 + [40] * 5,
            item_ids=[100] * 50 + [200] * 30 + [300] * 15 + [400] * 5,
        )
        assert_csv_format(df)

        result = preprocessor.replace_infrequent_values(df.copy())

        assert preprocessor.isInitialized is True
        expected_shops = set(top_90(df["shop_id"])["shop_id"])
        expected_items = set(top_90(df["item_id"])["item_id"])
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
        self, sales_df_factory, assert_csv_format, top_90
    ):
        """Work when initialize() has already been called on the same table.

        Rare shop_id and item_id values should still become 'other' using
        the top-value lists stored during initialize().
        """
        preprocessor = SalesDataPreprocessor()
        df = sales_df_factory(
            shop_ids=[10] * 50 + [20] * 30 + [30] * 15 + [40] * 5,
            item_ids=[100] * 50 + [200] * 30 + [300] * 15 + [400] * 5,
        )
        assert_csv_format(df)

        preprocessor.initialize(df)
        assert preprocessor.isInitialized is True

        result = preprocessor.replace_infrequent_values(df.copy())

        expected_shops = set(top_90(df["shop_id"])["shop_id"])
        expected_items = set(top_90(df["item_id"])["item_id"])
        result_shops = {int(value) for value in set(result["shop_id"]) - {"other"}}
        result_items = {int(value) for value in set(result["item_id"]) - {"other"}}
        assert result_shops == expected_shops
        assert result_items == expected_items
        assert "other" in set(result["shop_id"])
        assert "other" in set(result["item_id"])

    def test_initialized_and_uninitialized_paths_agree_on_the_same_frame(
        self, sales_df_factory
    ):
        """Initialized and uninitialized calls return the same table.

        If both paths see the same sales data, the replaced shop_id and
        item_id columns should match.
        """
        df = sales_df_factory(
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
        self, sales_df_factory, assert_csv_format
    ):
        """After initialize(), keep using that table's top IDs, not later counts.

        IDs that were rare when initialize() ran stay 'other' even if they
        are common in a later table.
        """
        preprocessor = SalesDataPreprocessor()
        train_df = sales_df_factory(
            shop_ids=[10] * 50 + [20] * 30 + [30] * 15 + [40] * 5,
            item_ids=[100] * 91 + [200] * 9,
        )
        later_df = sales_df_factory(
            shop_ids=[40] * 80 + [10] * 20,
            item_ids=[200] * 80 + [100] * 20,
        )
        assert_csv_format(later_df)

        preprocessor.initialize(train_df)
        result = preprocessor.replace_infrequent_values(later_df.copy())

        stored_shops = set(preprocessor.top_vals_by_col["shop_id"]["shop_id"])
        stored_items = set(preprocessor.top_vals_by_col["item_id"]["item_id"])
        assert 40 not in stored_shops
        assert list(result["shop_id"]).count("other") == 80
        assert {int(value) for value in set(result["shop_id"]) - {"other"}} <= stored_shops
        assert {int(value) for value in set(result["item_id"]) - {"other"}} <= stored_items

    def test_works_on_a_sample_of_sales_train_cnt_abs(
        self, csv_sample, assert_csv_format, top_90, expected_columns
    ):
        """Run the full replacement on a real sample of sales_train_cnt_abs.csv.

        Rare shop_id and item_id values become 'other'; date, month,
        price, and sales columns stay the same.
        """
        preprocessor = SalesDataPreprocessor()
        assert preprocessor.isInitialized is False
        assert_csv_format(csv_sample)

        result = preprocessor.replace_infrequent_values(csv_sample.copy())

        assert preprocessor.isInitialized is True
        for col in CAT_FEATURES:
            expected_ids = set(top_90(csv_sample[col])[col])
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
        assert list(result.columns) == expected_columns
