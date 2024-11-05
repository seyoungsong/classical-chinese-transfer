import sys
from importlib import reload

import typer
from loguru import logger
from pandarallel import pandarallel
from rich import pretty
from tqdm import tqdm

import src.corpus.ajd.root as sroot
from src import utils


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

    # 기사 번호 (cko)
    df1 = df[df["meta.data_id.cko"].notna()].reset_index(drop=True)
    df1["temp_id.cko"] = df1["meta.data_id.cko"].apply(lambda x: str(x).split("_")[-1])
    df1["temp_id.cko"].value_counts()
    df1["temp_id.cko"] = df1["temp_id.cko"].apply(lambda x: int(x[:4]))
    df1["temp_id.cko"].value_counts()

    # 기사 번호 (hj)
    df1["temp_id.hj"] = df1["meta.data_id.hj"].apply(lambda x: str(x).split("_")[-1])
    df1["temp_id.hj"].value_counts()
    df1["temp_id.hj"] = df1["temp_id.hj"].apply(lambda x: int(x))
    df1["temp_id.hj"].value_counts()
    df1.sample(1).iloc[0].to_dict()

    # diff
    # hj 1개가 쪼개져서 cko 2개로 나눠짐
    # https://db.itkc.or.kr/dir/item?itemId=JR#/dir/node?dataId=ITKC_JR_D0_A09_12A_14A_00030
    # https://db.itkc.or.kr/dir/item?itemId=JR#/dir/node?dataId=ITKC_JR_D0_A13_07A_17A_00040
    idx = df1["temp_id.cko"] != df1["temp_id.hj"]
    df2 = df1[idx].reset_index(drop=True)
    df2.sample(1).iloc[0].to_dict()

    # drop all day that has bad alignment
    df["parent_id"] = df["meta.data_id.hj"].apply(
        lambda x: str(x).rsplit("_", maxsplit=1)[0]
    )
    df2["parent_id"] = df2["meta.data_id.hj"].apply(
        lambda x: str(x).rsplit("_", maxsplit=1)[0]
    )
    idx = df["parent_id"].isin(df2["parent_id"])
    cols = [c for c in df.columns if c.endswith(".cko")]
    df.loc[idx, cols].sample(1).iloc[0].to_dict()
    df.loc[idx, cols] = None
    df.drop(columns="parent_id", inplace=True)

    # check en
    df1 = df[df["meta.data_id.en"].notna()].reset_index(drop=True)
    idx = df1["meta.data_id.en"].str[1:] != df1["meta.data_id.hj"].str[1:]
    assert idx.sum() == 0, "en != hj"

    # blank
    df = utils.replace_blank_to_none(df)

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
    gen_filter_file()  # 556.3M, 8324f4ab, 413323


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
            # python -m src.corpus.ajd.filter
            typer.run(main)
