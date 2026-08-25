"""Tests for replace_vals_by_col: swapping rare IDs for the label 'other'."""

import pandas as pd
import pytest


class TestReplaceValsByCol:
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
