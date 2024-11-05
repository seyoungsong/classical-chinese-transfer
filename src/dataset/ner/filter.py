import sys
from importlib import reload

import pandas as pd
import typer
from loguru import logger
from pandarallel import pandarallel
from rich import pretty
from tqdm import tqdm

import src.dataset.ner.root as sroot
from src import utils


def dedup_each_split(df: pd.DataFrame) -> pd.DataFrame:
    df_list = [
        df_group.drop_duplicates(subset=["text_xml"])
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
    merge_df = pd.merge(train_df, test_df, on=["text_xml"], how="left", indicator=True)
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
    cols = ["text_xml", "key2"]
    train_df = df[df["split"] == "train"][cols].reset_index(drop=True)
    valid_df = df[df["split"] == "valid"][cols].reset_index(drop=True)
    test_df = df[df["split"] == "test"][cols].reset_index(drop=True)

    # dedup
    test_df.drop_duplicates(subset=["text_xml"], inplace=True, ignore_index=True)

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
    cols = ["text_xml", "key2"]
    train_df = df[df["split"] == "train"][cols].reset_index(drop=True)
    valid_df = df[df["split"] == "valid"][cols].reset_index(drop=True)

    # dedup
    valid_df.drop_duplicates(subset=["text_xml"], inplace=True, ignore_index=True)

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
    df0 = utils.read_df(sroot.FORMAT_PQ)
    df = df0.copy()
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
    result = (
        (df["split"].value_counts() / df0["split"].value_counts() * 100)
        .round(2)
        .sort_index()
    )
    logger.debug(result)
    result
    """
split
test       93.94
test2     100.00
train      89.54
train2    100.00
valid      92.01
valid2    100.00
    """

    # length filtering
    # (we skip because hface trainer can truncate long sentences)

    # check
    if 0:
        idx = df0["key2"].isin(df["key2"])
        df_bad = df0[~idx].reset_index(drop=True)
        df_bad["split"].value_counts()
        df_bad.groupby(["meta.corpus", "split"]).size()
        df_bad.sample(1).iloc[0].to_dict()
        df_bad.sort_values(by=["text_xml"], inplace=True, ignore_index=True)
        utils.write_df(utils.TEMP_TSV, df_bad[["text_xml"]])

    # save
    utils.write_df2(sroot.FILTER_PQ, df)


def main() -> None:
    # dedup, prevent leakage
    gen_filter_file()  # 339.3M, 45bc7b38, 410382


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
            # python -m src.dataset.ner.filter
            typer.run(main)
