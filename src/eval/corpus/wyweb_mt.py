import sys
from importlib import reload
from typing import Any

import humanize
import pandas as pd
import typer
from loguru import logger
from pandarallel import pandarallel
from rich import pretty
from tqdm import tqdm

import src.crawl.wyweb_mt.root
import src.eval.corpus.root as sroot
import src.tool.eval as etool
from src import utils

NAME = "wyweb_mt"


def gen_stat() -> None:
    # read
    df_orig_both = utils.read_df(src.crawl.wyweb_mt.root.FORMAT2_PQ)
    df_orig_both.sample(1).iloc[0].to_dict()

    # clean
    df_orig_both = utils.replace_blank_to_none(df_orig_both)
    df_orig_both.isna().sum()[df_orig_both.isna().sum() > 0]

    # filter
    df_hj = df_orig_both.reset_index(drop=True)
    df_hj.rename(columns={"text_cc": "text"}, inplace=True)
    df_hj.reset_index(drop=True, inplace=True)
    df_hj.sample(1).iloc[0].to_dict()

    # output
    d1: dict[str, Any] = {}

    # avg_char
    d1["avg_char"] = df_hj["text"].str.len().mean()
    d1["total_char"] = df_hj["text"].str.len().sum()

    # num_sample
    d1["num_sample"] = len(df_hj)

    # avg_token
    if "token_len" not in df_hj.columns:
        df_hj["token_len"] = df_hj["text"].progress_apply(utils.num_tiktoken)
    d1["avg_token"] = df_hj["token_len"].mean()
    d1["total_token"] = df_hj["token_len"].sum()

    # ratio_mt
    d1["ratio_mt"] = (
        df_hj["text"].notnull().sum() / df_hj["text_zh"].notnull().sum() * 100
    )
    if 0:
        vc = df_hj["text"].value_counts()
        vc2 = vc[vc > 10].reset_index().to_dict(orient="records")
        utils.write_json(utils.TEMP_JSON, vc2)
        utils.open_code(utils.TEMP_JSON)

    # ratio_ner
    d1["ratio_ner"] = "N/A"

    # ratio_punc
    if "is_punc" not in df_hj.columns:
        df_hj["is_punc"] = df_hj["text"].parallel_apply(utils.is_punctuated_unicode)
    d1["ratio_punc"] = df_hj["is_punc"].mean() * 100

    # year
    d1["year"] = "N/A"

    # flatten
    d1 = pd.json_normalize(d1).to_dict(orient="records")[0]  # type: ignore

    # human
    for k, v in list(d1.items()):
        if isinstance(v, int) and "_humanize" not in k:
            d1[f"{k}_humanize"] = humanize.intword(v)

    # sort
    d1 = utils.sort_dict(d1)

    # write
    fname = sroot.RESULT_DIR / NAME / "stat.json"
    fname.parent.mkdir(parents=True, exist_ok=True)
    utils.write_json(fname, d1)

    # gen_char
    etool.report_char_punc_freq(texts=df_hj["text"], output_dir=sroot.RESULT_DIR / NAME)  # type: ignore


def main() -> None:
    gen_stat()


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
            # python -m src.eval.corpus.wyweb_mt
            typer.run(main)
