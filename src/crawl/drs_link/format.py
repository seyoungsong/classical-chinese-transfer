import sys
from importlib import reload

import typer
from loguru import logger
from rich import pretty
from tqdm import tqdm

import src.crawl.drs_link.root as sroot
from src import utils


def gen_format_file() -> None:
    # init
    tqdm.pandas()

    # read
    df = utils.read_df(sroot.LV1A_PQ)
    df.sample(1).iloc[0].to_dict()

    # rename cols
    {k: k for k in df.columns}
    rcols = {
        "data_id": "data_id",
        "text": "label",
        "url": "meta.url",
    }
    df.rename(columns=rcols, inplace=True)

    # fix
    df.sample(1).iloc[0].to_dict()
    vc = df["label"].value_counts(dropna=False)
    vc[vc > 3]

    #
    df["label"].isna().sum()
    df["label"].replace("Empty!", "", inplace=True)

    # parse
    if 0:
        s1: str = df["label"].sample(1).iloc[0]
        s1.split("http://sjw.history.go.kr/id/")[-1]
    df["label"] = df["label"].progress_apply(
        lambda x: str(x).split("http://sjw.history.go.kr/id/")[-1]
    )
    df["label"].value_counts().value_counts().sort_index()

    # sort cols
    c1 = [c for c in df.columns if not c.startswith("meta")]
    c2 = [c for c in df.columns if c.startswith("meta")]
    cols = sorted(c1) + sorted(c2)
    df = df[cols].reset_index(drop=True)
    df.sample(1).iloc[0].to_dict()

    # write
    df.sample(1).iloc[0].to_dict()
    utils.write_df2(sroot.FORMAT_PQ, df)
    logger.debug(len(df))


def main() -> None:
    # rename and sort cols
    gen_format_file()  # 2.7M, 56852c89, 552965


if __name__ == "__main__":
    if hasattr(sys, "ps1"):
        pretty.install()
        reload(utils)
        reload(sroot)
    else:
        with logger.catch(onerror=lambda _: sys.exit(1)):
            # python -m src.crawl.drs_link.format
            typer.run(main)
