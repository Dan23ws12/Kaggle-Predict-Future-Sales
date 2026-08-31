import os
import pandas as pd


def _item_cnt_lookup(preds: pd.DataFrame) -> pd.Series:
    """One item_cnt_month per shop_id, item_id (string keys)."""
    return (
        preds[["shop_id", "item_id", "item_cnt_month"]]
        .drop_duplicates(subset=["shop_id", "item_id"], keep="first")
        .set_index(["shop_id", "item_id"])["item_cnt_month"]
    )


def _reindex_item_cnt(lookup: pd.Series, shop: pd.Series, item: pd.Series) -> pd.Series:
    keys = pd.MultiIndex.from_arrays([shop.to_numpy(), item.to_numpy()])
    return pd.Series(lookup.reindex(keys).to_numpy(), index=shop.index)


def write_submission_csv(sales_df: pd.DataFrame, filename: str) -> pd.DataFrame:
    """
    Maps predicted item_cnt_month values onto test.csv IDs and writes a
    sample_submission-format CSV to the submission data folder.

    Rare shop_id or item_id values that were collapsed to "other" in sales_df
    are filled with the matching "other" pair: missing item_id uses
    (shop_id, "other"); missing shop_id uses ("other", item_id). Remaining
    gaps use ("other", "other"), then 0.
    """
    submit_ids = pd.read_csv(os.getenv("ORIGINAL_DATA_PATH") + "/test.csv")
    preds = sales_df[["shop_id", "item_id", "item_cnt_month"]].copy()
    preds["shop_id"] = preds["shop_id"].astype(str)
    preds["item_id"] = preds["item_id"].astype(str)
    lookup = _item_cnt_lookup(preds)

    shop = submit_ids["shop_id"].astype(str)
    item = submit_ids["item_id"].astype(str)
    known_shops = set(preds["shop_id"].unique())
    known_items = set(preds["item_id"].unique())

    item_cnt = _reindex_item_cnt(lookup, shop, item)

    item_missing = ~item.isin(known_items)
    shop_missing = ~shop.isin(known_shops)

    need = item_cnt.isna() & item_missing
    item_cnt = item_cnt.where(~need, _reindex_item_cnt(lookup, shop, pd.Series("other", index=item.index)))

    need = item_cnt.isna() & shop_missing
    item_cnt = item_cnt.where(~need, _reindex_item_cnt(lookup, pd.Series("other", index=shop.index), item))

    need = item_cnt.isna()
    item_cnt = item_cnt.where(~need, _reindex_item_cnt(lookup, shop, pd.Series("other", index=item.index)))
    need = item_cnt.isna()
    item_cnt = item_cnt.where(~need, _reindex_item_cnt(lookup, pd.Series("other", index=shop.index), item))
    need = item_cnt.isna()
    item_cnt = item_cnt.where(
        ~need,
        _reindex_item_cnt(
            lookup,
            pd.Series("other", index=shop.index),
            pd.Series("other", index=item.index),
        ),
    )

    submission = pd.DataFrame({
        "ID": submit_ids["ID"],
        "item_cnt_month": item_cnt.fillna(0).to_numpy(),
    })
    out_path = os.path.join(os.getenv("SUBMISSION_DATA_PATH"), os.path.basename(filename))
    submission.to_csv(out_path, index=False)
    return submission
