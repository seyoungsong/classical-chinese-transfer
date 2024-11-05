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

import src.corpus.ocdb_mt.root
import src.crawl.klc_hj.root
import src.crawl.klc_hj_ko.root
import src.crawl.ocdb_cc_ko.root
import src.eval.corpus.root as sroot
import src.tool.eval as etool
from src import utils

NAME = "ocdb"


def gen_stat() -> None:
    # read
    df_orig = utils.read_df(src.corpus.ocdb_mt.root.SPLIT_PQ)
    df_orig.sample(1).iloc[0].to_dict()

    # clean
    df_orig = utils.replace_blank_to_none(df_orig)
    df_orig.isna().sum()[df_orig.isna().sum() > 0]

    # filter
    df_hj = df_orig.copy()
    df_hj.rename(columns={"text.cc": "text"}, inplace=True)
    df_hj.dropna(subset=["text"], inplace=True)
    df_hj.sample(1).iloc[0].to_dict()

    # filter
    df_ko = df_orig.copy()
    df_ko.rename(columns={"text.ko": "text"}, inplace=True)
    df_ko.dropna(subset=["text"], inplace=True)
    df_ko.sample(1).iloc[0].to_dict()

    # output
    d1: dict[str, Any] = {}

    # avg_char
    d1["avg_char"] = df_hj["text"].str.len().mean()
    d1["total_char"] = df_hj["text"].str.len().sum()

    # num_sample
    d1["num_sample"] = len(df_hj)

    # avg_token
    if "token_len" not in df_hj.columns:
        df_hj["token_len"] = df_hj["text"].parallel_apply(utils.num_tiktoken)
    d1["avg_token"] = df_hj["token_len"].mean()
    d1["total_token"] = df_hj["token_len"].sum()

    # ratio_mt
    d1["ratio_mt"] = 100.0
    if 0:
        vc = df_ko["text"].value_counts()
        vc2 = vc[vc > 10].reset_index().to_dict(orient="records")
        utils.write_json(utils.TEMP_JSON, vc2)
        utils.open_code(utils.TEMP_JSON)

    # ratio_ner
    d1["ratio_ner"] = 0.0

    # ratio_punc
    if "is_punc" not in df_hj.columns:
        df_hj["is_punc"] = df_hj["text"].parallel_apply(
            lambda x: utils.is_punctuated_unicode(utils.remove_whites(x))
            and (("," in x) or ("." in x) or ("。" in x))
        )
    d1["ratio_punc"] = df_hj["is_punc"].mean() * 100
    if 0:
        df_hj[df_hj["is_punc"]].sample(1).iloc[0].to_dict()["text"]
        df_hj[~df_hj["is_punc"]].sample(1).iloc[0].to_dict()["text"]

    # year
    d1["year"] = "N/A"
    df_hj.sample(1).iloc[0].to_dict()
    books: list[str] = sorted(df_hj["meta.bookname"].unique())
    books = sorted(set([s.split("(")[0] for s in books]))
    fname = sroot.RESULT_DIR / NAME / "books.json"
    fname.parent.mkdir(parents=True, exist_ok=True)
    utils.write_json(fname, books)
    utils.write_str(fname.with_suffix(".txt"), "\n".join(books))

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
            # python -m src.eval.corpus.ocdb
            typer.run(main)
