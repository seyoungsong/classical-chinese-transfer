import sys
from importlib import reload

import pandas as pd
import typer
from loguru import logger
from pandarallel import pandarallel
from rich import pretty
from tqdm import tqdm

import src.dataset.mt_h2ke.root as sroot
from src import utils


def dedup_each_split(df: pd.DataFrame) -> pd.DataFrame:
    df_list = [
        df_group.drop_duplicates(subset=["text.src", "text.tgt"])
        for _, df_group in df.groupby("split")
    ]
    df1 = pd.concat(df_list, ignore_index=True)
    df1.sort_values(by="key2", inplace=True, ignore_index=True)

    # check
    vc1 = df1["split"].value_counts().sort_index()
    vc0 = df["split"].value_counts().sort_index()
    logger.debug((vc1 / vc0 * 100).sort_index().round(1))

    return df1


def find_dup(train_df: pd.DataFrame, test_df: pd.DataFrame) -> pd.DataFrame:
    merge_df = pd.merge(
        train_df, test_df, on=["text.src", "text.tgt"], how="left", indicator=True
    )
    merge_df["_merge"].value_counts()
    merge_df = merge_df[merge_df["_merge"] == "both"].reset_index(drop=True)
    merge_df.rename(columns={"key2_x": "key2"}, inplace=True)
    if 0:
        merge_df.sample(1).iloc[0].to_dict()
    return merge_df


def remove_test_from_train_valid(df: pd.DataFrame) -> pd.DataFrame:
    # check
    df["split"].value_counts()
    assert df["key2"].is_unique, "key2 is not unique"

    # split
    cols = ["text.src", "text.tgt", "key2"]
    train_df = df[df["split"] == "train"][cols].reset_index(drop=True)
    valid_df = df[df["split"] == "valid"][cols].reset_index(drop=True)
    test_df = df[df["split"] == "test"][cols].reset_index(drop=True)

    # dedup
    test_df.drop_duplicates(
        subset=["text.src", "text.tgt"], inplace=True, ignore_index=True
    )

    # find dup
    train_dup = find_dup(train_df=train_df, test_df=test_df)
    valid_dup = find_dup(train_df=valid_df, test_df=test_df)
    final_dup = pd.concat([train_dup, valid_dup], ignore_index=True)

    # check
    assert final_dup["key2"].is_unique
    idx = df["key2"].isin(final_dup["key2"])
    logger.debug(df[idx].groupby("split").size())

    # remove
    df = df[~idx].reset_index(drop=True)

    return df


def remove_valid_from_train(df: pd.DataFrame) -> pd.DataFrame:
    # check
    df["split"].value_counts()
    assert df["key2"].is_unique, "key2 is not unique"

    # split
    cols = ["text.src", "text.tgt", "key2"]
    train_df = df[df["split"] == "train"][cols].reset_index(drop=True)
    valid_df = df[df["split"] == "valid"][cols].reset_index(drop=True)

    # dedup
    valid_df.drop_duplicates(
        subset=["text.src", "text.tgt"], inplace=True, ignore_index=True
    )

    # find dup
    final_dup = find_dup(train_df=train_df, test_df=valid_df)

    # check
    assert final_dup["key2"].is_unique
    idx = df["key2"].isin(final_dup["key2"])
    logger.debug(df[idx].groupby("split").size())

    # remove
    df = df[~idx].reset_index(drop=True)

    return df


def gen_filter_file() -> None:
    # read
    df0 = utils.read_df(sroot.CONCAT_PQ)
    df = df0.copy()
    df.sample(1).iloc[0].to_dict()

    # merge url cols
    ucols: list[str] = [c for c in df.columns if "url" in c and c != "meta.url"]
    ucols = df[ucols].notna().sum().sort_values().index.to_list()
    ucols1 = [c for c in ucols if "data_url" not in c]
    ucols2 = [c for c in ucols if "data_url" in c]
    ucols = ucols1 + ucols2
    df["meta.url"] = None
    for c in tqdm(ucols):
        df["meta.url"] = df["meta.url"].combine_first(df[c])
    df["meta.url"].notna().mean() * 100  # 84.9
    if 0:
        d1 = (
            df.groupby(["meta.corpus", "lang.src", "lang.tgt"])
            .sample(1)
            .to_dict(orient="records")
        )
        utils.write_json(utils.TEMP_JSON, d1)
        utils.open_code(utils.TEMP_JSON)

    # drop cols
    keep = ["meta.corpus", "meta.lang.ko", "meta.url"]
    cols = [c for c in df.columns if c.startswith("meta") and c not in keep]
    df.drop(columns=cols, inplace=True)

    # sort cols
    c1 = [c for c in df.columns if not c.startswith("meta")]
    c2 = [c for c in df.columns if c.startswith("meta")]
    cols = sorted(c1) + sorted(c2)
    df = df[cols].reset_index(drop=True)
    df.sample(1).iloc[0].to_dict()

    # dedup each split, remove test from train & valid, remove valid from train
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    df = dedup_each_split(df)
    df = remove_test_from_train_valid(df)
    df = remove_valid_from_train(df)
    df.sort_values(by="key2", inplace=True, ignore_index=True)

    # check
    df0["split"].value_counts() / len(df0) * 100
    df["split"].value_counts() / len(df) * 100
    df["split"].value_counts() / df0["split"].value_counts() * 100

    # check
    if 0:
        idx = df0["key2"].isin(df["key2"])
        df_bad = df0[~idx].reset_index(drop=True)
        df_bad["split"].value_counts()
        df_bad.groupby(["meta.corpus", "split"]).size()
        df_bad.sample(1).iloc[0].to_dict()
        df_bad.sort_values(by=["text.src", "text.tgt"], inplace=True, ignore_index=True)
        utils.write_df(utils.TEMP_TSV, df_bad[["text.src", "text.tgt"]])

    # save
    utils.write_df2(sroot.FILTER_PQ, df)


def main() -> None:
    # dedup, prevent leakage
    gen_filter_file()  # 608.1M, d33ecb5e, 2712165


if __name__ == "__main__":
    tqdm.pandas()
    pandarallel.initialize(progress_bar=True)
    #
    if hasattr(sys, "ps1"):
        pretty.install()
        reload(utils)
        reload(sroot)
    else:
        with logger.catch(onerror=lambda _: sys.exit(1)):
            # python -m src.dataset.mt_h2ke.filter
            typer.run(main)
