import sys
from importlib import reload
from typing import Any

import humanize
import typer
from loguru import logger
from rich import pretty
from tqdm import tqdm

import src.crawl.wyweb_ner.root as sroot
from src import utils


def gen_stat_json() -> None:
    # init
    tqdm.pandas()

    # read
    df_orig = utils.read_df(sroot.FORMAT2_PQ)
    df_orig.sample(1).iloc[0].to_dict()

    # prep
    d1: dict[Any, Any] = {}

    # char
    cols = ["text"]
    df_hj = (
        df_orig[cols].stack(dropna=False).reset_index(drop=True).to_frame("text").copy()  # type: ignore
    )
    df_hj.isna().mean()
    df_hj.dropna(subset=["text"], inplace=True)
    df_hj["char_len"] = df_hj["text"].str.len()
    d2 = df_hj["char_len"].agg(["count", "mean", "median", "sum"]).to_dict()
    d2["sum_human"] = humanize.intword(round(d2["sum"]), format="%.0f")
    d1["char"] = d2

    # token
    if "token_len" not in df_hj.columns:
        df_hj["token_len"] = df_hj["text"].progress_apply(utils.num_tiktoken)
    d1["avg_token"] = df_hj["token_len"].mean()
    d1["total_token"] = df_hj["token_len"].sum()
    d1["total_token_human"] = humanize.intword(d1["total_token"], format="%.0f")

    sroot.STAT_JSON.parent.mkdir(parents=True, exist_ok=True)
    utils.write_json(sroot.STAT_JSON, d1)


def main() -> None:
    # check basic stats
    gen_stat_json()


if __name__ == "__main__":
    if hasattr(sys, "ps1"):
        pretty.install()
        reload(utils)
        reload(sroot)
    else:
        with logger.catch(onerror=lambda _: sys.exit(1)):
            # python -m src.crawl.wyweb_ner.check
            typer.run(main)
