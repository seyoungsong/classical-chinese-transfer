import sys
from importlib import reload

import typer
from loguru import logger
from rich import pretty
from tqdm import tqdm

import src.crawl.ocdb_cc_ko.root as sroot
from src import utils


def gen_format2_file() -> None:
    # read
    df = utils.read_df(sroot.FORMAT_PQ)
    df.sample(1).iloc[0].to_dict()

    # empty
    df = utils.replace_blank_to_none(df)
    df.info()

    # sort columns
    c1 = [c for c in df.columns if not c.startswith("meta")]
    c2 = [c for c in df.columns if c.startswith("meta")]
    cols = sorted(c1) + sorted(c2)
    df = df[cols].reset_index(drop=True)
    df.sample(1).iloc[0].to_dict()

    # write
    utils.write_df2(sroot.FORMAT2_PQ, df)


def main() -> None:
    # change format
    gen_format2_file()  # 29.6M, c8253283, 28341


if __name__ == "__main__":
    tqdm.pandas()
    #
    if hasattr(sys, "ps1"):
        pretty.install()
        reload(utils)
        reload(sroot)
    else:
        with logger.catch(onerror=lambda _: sys.exit(1)):
            # python -m src.crawl.ocdb_cc_ko.format2
            typer.run(main)
