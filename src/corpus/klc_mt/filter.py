import sys
from importlib import reload

import typer
from loguru import logger
from pandarallel import pandarallel
from rich import pretty
from tqdm import tqdm

import src.corpus.klc_mt.root as sroot
from src import utils


def gen_filter_file() -> None:
    # read
    df0 = utils.read_df(sroot.SPLIT_PQ)
    df = df0.copy()
    df.sample(1).iloc[0].to_dict()

    # blank
    df = utils.replace_blank_to_none(df)

    # check
    df.isna().sum()[df.isna().sum() > 0]

    # fillna: meta.data_copyright.hj
    idx = df["meta.data_copyright.hj"].isna()
    idx.sum()
    df[idx].sample(1).iloc[0].to_dict()
    df.loc[idx, "meta.data_copyright.hj"] = "N/A"

    # fillna: meta.data_dci.ko
    idx = df["meta.data_dci.ko"].isna()
    idx.sum()
    df[idx].sample(1).iloc[0].to_dict()
    df.loc[idx, "meta.data_dci.ko"] = "N/A"

    # check: null diff -> table or image (drop)
    idx = df["text.ko"].isna() != df["text.hj"].isna()
    idx.sum()  # 79
    df[idx].sample(1).iloc[0].to_dict()
    df = df[~idx].reset_index(drop=True)

    # check: both null
    # 이 자료는 규장각(奎章閣) 소장 한글필사본입니다. 번역문 및 원문이미지 보기를 참고하시기 바랍니다.
    idx = df["text.ko"].isna() & df["text.hj"].isna()
    idx.sum()  # 287
    df[idx].sample(1).iloc[0].to_dict()
    df = df[~idx].reset_index(drop=True)

    # check
    df = utils.replace_blank_to_none(df)
    df.info()

    # sample
    if 0:
        df1 = df.dropna()
        x1 = df1.sample(1).iloc[0].to_dict()
        utils.write_json(utils.TEMP_JSON, x1)
        utils.open_code(utils.TEMP_JSON)

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
    gen_filter_file()  # 613.9M, 43dd373b, 156836


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
            # python -m src.corpus.klc_mt.filter
            typer.run(main)
