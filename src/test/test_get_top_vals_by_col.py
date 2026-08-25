"""Tests for get_top_vals_by_col: which IDs count as the top 90%."""

import pandas as pd


class TestGetTopValsByCol:
    def test_returns_highest_frequency_values_until_90_percent_coverage(
        self, preprocessor, top_90
    ):
        """Keep the most common IDs until their counts cover 90% of rows.

        With counts 50, 30, 15, and 5, the first three IDs cover 95 rows
        and the rarest ID (count 5) must be left out.
        """
        shop_ids = [1] * 50 + [2] * 30 + [3] * 15 + [4] * 5
        df = pd.DataFrame({"shop_id": shop_ids})

        result = preprocessor.get_top_vals_by_col(df, "shop_id", increment=1)
        expected = top_90(df["shop_id"])

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
        self, preprocessor, top_90
    ):
        """Do not keep extra IDs after the 90% coverage target is met.

        Ten IDs that each appear once need only nine of them to cover 90%
        of the rows, so the last ID is excluded.
        """
        df = pd.DataFrame({"shop_id": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]})

        result = preprocessor.get_top_vals_by_col(df, "shop_id", increment=1)
        expected = top_90(df["shop_id"])

        pd.testing.assert_frame_equal(
            result.reset_index(drop=True), expected, check_dtype=False
        )
        assert len(result) == 9
        assert result["frequency"].sum() >= 0.9 * len(df)

    def test_increment_does_not_include_values_beyond_the_90_percent_cutoff(
        self, preprocessor, top_90
    ):
        """A larger lookup step still returns the same 90% cutoff, not extra IDs.

        The function can scan the frequency table in chunks (here, size 2)
        for speed, but the returned IDs must still be exactly those needed
        to reach 90% coverage.
        """
        shop_ids = [1] * 50 + [2] * 30 + [3] * 15 + [4] * 5
        df = pd.DataFrame({"shop_id": shop_ids})
        expected = top_90(df["shop_id"])

        result = preprocessor.get_top_vals_by_col(df, "shop_id", increment=2)

        pd.testing.assert_frame_equal(
            result.reset_index(drop=True), expected, check_dtype=False
        )
