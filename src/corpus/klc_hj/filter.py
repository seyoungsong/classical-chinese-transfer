import sys
from importlib import reload

import typer
from loguru import logger
from pandarallel import pandarallel
from rich import pretty
from tqdm import tqdm

import src.corpus.klc_hj.root as sroot
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

    # check: null
    idx = df["text.hj"].isna()
    idx.sum()  # 764
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
    gen_filter_file()  # 393.9M, d420392c, 652622


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
            # python -m src.corpus.klc_hj.filter
            typer.run(main)
