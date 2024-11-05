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

import src.crawl.ajd_en.root
import src.eval.corpus.root as sroot
import src.tool.eval as etool
from src import utils

NAME = "mt_en"


def get_ajd_en() -> pd.DataFrame:
    # read
    df = utils.read_df(src.crawl.ajd_en.root.FORMAT2_PQ)
    df.sample(1).iloc[0].to_dict()

    # filter
    df["lang"].value_counts()
    df = df[df["lang"] == "en"].reset_index(drop=True)

    # check
    assert df["text"].isna().sum() == 0

    return df


def get_df_ko() -> pd.DataFrame:
    # read
    df_list = [get_ajd_en()]

    # concat
    df_ko = pd.concat(df_list, ignore_index=True)
    df_ko.columns
    df_ko.sample(1).iloc[0].to_dict()

    # drop columns
    df_ko = df_ko[["text"]].reset_index(drop=True)

    # check
    assert df_ko["text"].isna().sum() == 0

    # check
    df_ko.sample(1).iloc[0].to_dict()

    return df_ko


def gen_stat() -> None:  # noqa: C901
    # read
    df_ko = get_df_ko()
    df_ko.sample(1).iloc[0].to_dict()

    # output
    d1: dict[str, Any] = {}

    # avg_char
    d1["avg_char"] = df_ko["text"].str.len().mean()
    d1["total_char"] = df_ko["text"].str.len().sum()

    # num_sample
    d1["num_sample"] = len(df_ko)

    # avg_token
    df_ko_sample = df_ko.sample(frac=0.05, random_state=42).reset_index(drop=True)
    df_ko_sample["token_len"] = df_ko_sample["text"].progress_apply(utils.num_tiktoken)
    d1["avg_token_frac"] = df_ko_sample["token_len"].mean()
    d1["avg_char_frac"] = df_ko_sample["text"].str.len().mean()
    d1["total_token_est"] = round(
        d1["num_sample"] * d1["avg_token_frac"] / d1["avg_char_frac"] * d1["avg_char"]
    )

    # ratio_mt
    if 0:
        d1["ratio_mt"] = len(df_ko) / len(df_ko) * 100
        vc = df_ko["text"].value_counts()
        vc2 = vc[vc > 10].reset_index().to_dict(orient="records")
        utils.write_json(utils.TEMP_JSON, vc2)
        utils.open_code(utils.TEMP_JSON)

    # ratio_ner
    if 0:
        if "is_ner" not in df_ko.columns:
            df_ko["is_ner"] = df_ko["text_xml"].parallel_apply(
                lambda x: f"<{utils.NER_PREF}" in str(x)
            )
        d1["ratio_ner"] = df_ko["is_ner"].mean() * 100

    # ratio_punc
    if 0:
        if "is_punc" not in df_ko.columns:
            df_ko["is_punc"] = df_ko["text"].parallel_apply(utils.is_punctuated_unicode)
        d1["ratio_punc"] = df_ko["is_punc"].mean() * 100

    # year
    if 0:
        pat = re.compile(r"\s*(\d{3,})년\s*")
        if 0:
            x1 = df_ko.sample(1).iloc[0].to_dict()
            x1["meta.data_date"]
            pat.search(x1["meta.data_date"]).group(1)
        if "year" not in df_ko.columns:
            df_ko["year"] = df_ko["meta.data_date"].parallel_apply(
                lambda x: pat.search(x).group(1) if pat.search(x) else None
            )
        vals = sorted(map(int, df_ko["year"].dropna().unique()))
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
    etool.report_char_punc_freq(texts=df_ko["text"], output_dir=sroot.RESULT_DIR / NAME)  # type: ignore


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
            # python -m src.eval.corpus.mt_en
            typer.run(main)
