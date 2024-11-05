import sys
from importlib import reload

import pandas as pd
import typer
from loguru import logger
from pandarallel import pandarallel
from rich import pretty
from tqdm import tqdm

import src.corpus.drri.root as sroot
from src import utils


# Function to check for mixed null and non-null values
def check_mixed_nulls(group: pd.Series) -> bool:  # type: ignore
    return group.isnull().any() and group.notnull().any()


def gen_filter_file() -> None:
    # read
    df0 = utils.read_df(sroot.SPLIT_PQ)
    df = df0.copy()
    df.sample(1).iloc[0].to_dict()

    # sample
    if 0:
        df1 = df.dropna()
        x1 = df1.sample(1).iloc[0].to_dict()
        utils.write_json(utils.TEMP_JSON, x1)
        utils.open_code(utils.TEMP_JSON)

    # check: meta.data_id.ko != meta.date_id2.hj
    df1 = df[df["meta.data_id.ko"].notna()].reset_index(drop=True)
    df1["temp1"] = df1["meta.data_id.ko"].apply(lambda x: str(x).split("_")[-1])
    df1["temp1"].str[:2].value_counts()
    df1["temp1"].str[4].value_counts()
    df1["temp1"] = df1["temp1"].apply(lambda x: int(x[:4]))
    df1.sample(1).iloc[0].to_dict()
    df1["temp2"] = df1["meta.date_id2.hj"].apply(lambda x: str(x).split("_")[-1])
    df1["temp2"] = df1["temp2"].apply(lambda x: int(str(x).split("/")[0]))
    idx = df1["temp1"] != df1["temp2"]
    assert idx.sum() == 0, "meta.data_id.ko != meta.date_id2.hj"

    # meta.date_id.hj 단위에서 전부 존재하거나 전부 존재하지 않아야 함
    df.sample(1).iloc[0].to_dict()
    mixed_null_groups = df.groupby("meta.date_id.hj")["meta.data_id.ko"].apply(
        check_mixed_nulls
    )
    bad_values = mixed_null_groups[mixed_null_groups].index.tolist()
    assert (
        len(bad_values) == 0
    ), "meta.date_id.hj 단위에서 전부 존재하거나 전부 존재하지 않아야 함"

    # meta.data_idx.hj가 0으로 시작하지 않는 경우 (한 날짜에 여러개의 page)
    df.sample(1).iloc[0].to_dict()
    df["meta.data_idx.hj"].str[:1].value_counts()
    idx = df["meta.data_idx.hj"].str[:1] != "0"
    df1 = df[idx].reset_index(drop=True)
    df1.sample(1).iloc[0].to_dict()
    df1["meta.data_id.ko"].value_counts(dropna=False)

    # blank
    df = utils.replace_blank_to_none(df)

    # check 'text.ko' null == 'text.hj' null
    df1 = df[df["meta.data_id.ko"].notna()].reset_index(drop=True)
    df1.sample(1).iloc[0].to_dict()
    idx1 = df1["text_body.hj"].isnull() != df1["text_body.ko"].isnull()
    idx1.sum()
    idx2 = df1["text_title.hj"].isnull() != df1["text_title.ko"].isnull()
    idx2.sum()
    idx = idx1 | idx2
    idx.sum()
    idx.mean() * 100
    df1[idx].sample(1).iloc[0].to_dict()
    bad_vals = df1[idx]["meta.date_id.hj"].unique()
    # replace None
    len(bad_vals)
    idx = df["meta.date_id.hj"].isin(bad_vals)
    idx.mean() * 100
    cols = [c for c in df.columns if c.endswith(".ko")]
    df.loc[idx, cols] = None

    # sort cols
    c1 = [c for c in df.columns if not c.startswith("meta")]
    c2 = [c for c in df.columns if c.startswith("meta")]
    cols = sorted(c1) + sorted(c2)
    df = df[cols].reset_index(drop=True)
    df.sample(1).iloc[0].to_dict()

    # save
    utils.write_df2(sroot.FILTER_PQ, df)


def main() -> None:
    # filter poorly aligned data
    gen_filter_file()  # 177.8M, 4c438bed, 367124


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
            # python -m src.corpus.drri.filter
            typer.run(main)
