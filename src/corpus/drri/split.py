import sys
from importlib import reload

import typer
from loguru import logger
from pandarallel import pandarallel
from rich import pretty
from tqdm import tqdm

import src.corpus.drri.root as sroot
import src.tool.data as dtool
from src import utils


def gen_split_file() -> None:
    # read
    df0 = utils.read_df(sroot.NORMAL_PQ)
    df = df0.copy()

    # sample
    if 0:
        df1 = df.dropna()
        x1 = df1.sample(1).iloc[0].to_dict()
        utils.write_json(utils.TEMP_JSON, x1)
        utils.open_code(utils.TEMP_JSON)

    # split
    df["split"] = df["key"].parallel_apply(
        lambda x: dtool.uid2split(uid=x, ratio=(0.8, 0.1, 0.1))
    )
    if 0:
        df["split"].value_counts() / len(df) * 100
        df["split"].value_counts()

    # sort cols
    c1 = [c for c in df.columns if not c.startswith("meta")]
    c2 = [c for c in df.columns if c.startswith("meta")]
    cols = sorted(c1) + sorted(c2)
    df = df[cols].reset_index(drop=True)
    df.sample(1).iloc[0].to_dict()

    # save
    utils.write_df2(sroot.SPLIT_PQ, df)


def main() -> None:
    # split train/valid/test by key
    gen_split_file()  # 208.3M, b74ffea1, 367124


if __name__ == "__main__":
    tqdm.pandas()
    pandarallel.initialize(progress_bar=True)
    #
    if hasattr(sys, "ps1"):
        pretty.install()
        reload(utils)
        reload(sroot)
        reload(dtool)
    else:
        with logger.catch(onerror=lambda _: sys.exit(1)):
            # python -m src.corpus.drri.split
            typer.run(main)
