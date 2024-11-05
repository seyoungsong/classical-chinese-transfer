import sys
from importlib import reload

import typer
from loguru import logger
from pandarallel import pandarallel
from rich import pretty
from tqdm import tqdm

import src.corpus.ocdb_mt.root as sroot
import src.tool.data as dtool
from src import utils


def gen_split_file() -> None:
    # read
    df = utils.read_df(sroot.NORMAL_PQ)

    # split
    assert df["key"].is_unique, "not unique key"
    df["split"] = df["key"].parallel_apply(
        lambda x: dtool.uid2split(uid=x, ratio=(0.8, 0.1, 0.1))
    )
    logger.debug(df["split"].value_counts())
    """
    train    19068
    test      2420
    valid     2307
    """

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
    gen_split_file()


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
            # python -m src.corpus.ocdb_mt.split
            typer.run(main)
