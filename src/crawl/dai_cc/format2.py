import sys
from importlib import reload

import typer
from loguru import logger
from pandarallel import pandarallel
from rich import pretty
from tqdm import tqdm

import src.crawl.dai_cc.root as sroot
from src import utils


def gen_format2_file() -> None:  # noqa: C901
    # init
    tqdm.pandas()
    pandarallel.initialize(progress_bar=True)

    # read
    df = utils.read_df(sroot.PARSE2_PQ)
    df.sample(1).iloc[0].to_dict()

    # squeeze
    df["text"] = df["text"].parallel_apply(utils.squeeze_whites)

    # sort columns
    c1 = [c for c in df.columns if not c.startswith("meta")]
    c2 = [c for c in df.columns if c.startswith("meta")]
    cols = sorted(c1) + sorted(c2)
    df = df[cols].reset_index(drop=True)

    # sort rows
    df.sort_values(by=["meta.data_id"], inplace=True, ignore_index=True)

    # write
    df.sample(1).iloc[0].to_dict()
    utils.write_df2(sroot.FORMAT2_PQ, df)
    logger.debug(f"len(df)={len(df)}")


def main() -> None:
    # convert text_xml to text and drop empty columns
    gen_format2_file()  # 2.5G, 49bfdd90, 15694


if __name__ == "__main__":
    if hasattr(sys, "ps1"):
        pretty.install()
        reload(utils)
        reload(sroot)
    else:
        with logger.catch(onerror=lambda _: sys.exit(1)):
            # python -m src.crawl.dai_cc.format2
            typer.run(main)
