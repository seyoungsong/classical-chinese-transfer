import sys
from importlib import reload

import typer
from loguru import logger
from pandarallel import pandarallel
from rich import pretty
from tqdm import tqdm

import src.corpus.ocdb_mt.root as sroot
from src import utils


def gen_format_file() -> None:
    # read
    df = utils.read_df(sroot.ALIGN_PQ)
    df.sample(1).iloc[0].to_dict()

    # check key
    assert df["meta.data_id"].is_unique, "key is not unique"
    df["key"] = df["meta.data_id"]

    # sort rows
    df.sort_values("key", inplace=True, ignore_index=True)

    # replace None
    df = utils.replace_blank_to_none(df)

    # sort cols
    c1 = [c for c in df.columns if not c.startswith("meta")]
    c2 = [c for c in df.columns if c.startswith("meta")]
    cols = sorted(c1) + sorted(c2)
    df = df[cols].reset_index(drop=True)
    df.sample(1).iloc[0].to_dict()

    # save
    utils.write_df2(sroot.FORMAT_PQ, df)


def main() -> None:
    # rename columns, add key, sort rows, sort cols
    gen_format_file()  # 23.0M, c92ca8c0, 23940


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
            # python -m src.corpus.ocdb_mt.format
            typer.run(main)
