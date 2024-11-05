import json
import sys
from importlib import reload

import typer
from loguru import logger
from rich import pretty
from tqdm import tqdm

import src.crawl.tower_mt.root as sroot
from src import utils


def gen_parse_file() -> None:
    # read
    df = utils.read_df(sroot.HFACE_PQ)
    df.info()
    df.sample(1).iloc[0].to_dict()

    # convert
    df["conversations"] = df["conversations"].progress_apply(
        lambda x: json.dumps(x.tolist(), ensure_ascii=False)
    )

    # check
    df.columns
    df1 = df[["lang", "split", "dataset", "task"]].reset_index(drop=True)
    df1.nunique()
    #
    df1["lang"].value_counts()  # ko, en, zh, mixed? 필터링할 것.
    df1["split"].value_counts()  # dev -> valid
    df1["dataset"].value_counts()  # 별도 필터링 없음.
    df1["task"].value_counts()  # 별도 필터링 없음.

    # save
    sroot.PARSE_PQ.parent.mkdir(exist_ok=True, parents=True)
    utils.write_df2(sroot.PARSE_PQ, df)
    logger.debug(f"len: {len(df)}")


def main() -> None:
    # basic parse raw html files into jsonl
    gen_parse_file()  # 468.9M, 0565f9dd, 637563


if __name__ == "__main__":
    tqdm.pandas()
    #
    if hasattr(sys, "ps1"):
        pretty.install()
        reload(utils)
        reload(sroot)
    else:
        with logger.catch(onerror=lambda _: sys.exit(1)):
            # python -m src.crawl.tower_mt.parse
            typer.run(main)
