import re
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

import src.crawl.drri_hj.root
import src.crawl.drri_ko.root
import src.eval.corpus.root as sroot
import src.tool.eval as etool
from src import utils

NAME = "drri"


def gen_stat() -> None:
    # read
    df_orig_hj = utils.read_df(src.crawl.drri_hj.root.FORMAT2_PQ)
    df_orig_hj.sample(1).iloc[0].to_dict()

    # read
    df_orig_ko = utils.read_df(src.crawl.drri_ko.root.FORMAT2_PQ)
    df_orig_ko.sample(1).iloc[0].to_dict()

    # clean
    df_orig_hj = utils.replace_blank_to_none(df_orig_hj)
    df_orig_hj.isna().sum()[df_orig_hj.isna().sum() > 0]

    # clean
    df_orig_ko = utils.replace_blank_to_none(df_orig_ko)
    df_orig_ko.isna().sum()[df_orig_ko.isna().sum() > 0]

    # filter
    cols = ["text_body", "text_title"]
    df_hj = df_orig_hj[cols].stack(dropna=True).reset_index(drop=True).to_frame("text")  # type: ignore

    # filter
    cols = ["text_body_xml", "text_title_xml"]
    df_hj_xml = df_orig_hj[cols].stack(dropna=True).reset_index(drop=True).to_frame("text_xml")  # type: ignore

    # filter
    df_orig_ko.sample(1).iloc[0].to_dict()
    cols = ["text", "meta.elem_title"]
    df_ko = df_orig_ko[cols].stack(dropna=True).reset_index(drop=True).to_frame("text")  # type: ignore

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
    d1["ratio_mt"] = len(df_ko) / len(df_hj) * 100
    if 0:
        vc = df_ko["text"].value_counts()
        vc2 = vc[vc > 10].reset_index().to_dict(orient="records")
        utils.write_json(utils.TEMP_JSON, vc2)
        utils.open_code(utils.TEMP_JSON)

    # ratio_ner
    if "is_ner" not in df_hj_xml.columns:
        df_hj_xml["is_ner"] = df_hj_xml["text_xml"].parallel_apply(
            lambda x: f"<{utils.NER_PREF}" in str(x)
        )
    d1["ratio_ner"] = df_hj_xml["is_ner"].mean() * 100

    # ratio_punc
    if "is_punc" not in df_hj.columns:
        df_hj["is_punc"] = df_hj["text"].parallel_apply(utils.is_punctuated_unicode)
    d1["ratio_punc"] = df_hj["is_punc"].mean() * 100

    # year
    pat = re.compile(r"\(\s*(\d+)\s*\)")
    if 0:
        x1 = df_orig_hj.sample(1).iloc[0].to_dict()
        x1["meta.data_date"]
        pat.search(x1["meta.data_date"]).group(1)
    if "year" not in df_orig_hj.columns:
        df_orig_hj["year"] = df_orig_hj["meta.data_date"].parallel_apply(
            lambda x: pat.search(x).group(1) if pat.search(x) else None  # type: ignore
        )
    vals = sorted(map(int, df_orig_hj["year"].dropna().unique()))
    d1["year"] = ", ".join(map(str, vals))

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
            # python -m src.eval.corpus.drri
            typer.run(main)
