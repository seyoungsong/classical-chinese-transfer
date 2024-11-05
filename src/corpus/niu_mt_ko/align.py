import sys
from importlib import reload

import pandas as pd
import typer
from loguru import logger
from pandarallel import pandarallel
from rich import pretty
from tqdm import tqdm

import src.corpus.niu_mt.root
import src.corpus.niu_mt_ko.root as sroot
import src.eval.mt_aug.root
from src import utils


def fix_aug(df_aug: pd.DataFrame) -> pd.DataFrame:
    # copy
    df = df_aug.copy()
    df.sample(1).iloc[0].to_dict()

    # check korean
    if 0:
        vc = df["pred.content"].str[:3].value_counts()
        vc = df["pred.content"].str[-3:].value_counts()
        vc = vc[vc > 10]
        utils.write_json(utils.TEMP_JSON, vc.to_dict())
        utils.open_code(utils.TEMP_JSON)

        # "Korean:"
        idx = df["pred.content"].str.startswith("Kor")
        idx.sum()
        df1 = df[idx].reset_index(drop=True)
        df1.sample(1).iloc[0].to_dict()
        df1["pred.content"].str.replace(r"^[Kk]orean:\s*", "", regex=True)

        # English
        idx = df["pred.content"].str.match(r"^[A-Za-z]")
        idx.sum()
        idx.mean() * 100
        df1 = df[idx].reset_index(drop=True)
        df1.sample(1).iloc[0].to_dict()["pred.content"]
        df1["pred.content"].str.replace(r"^[Kk]orean:\s*", "", regex=True)

    # fix korean
    df["pred.content"] = df["pred.content"].str.replace(
        r"^[Kk]orean:\s*", "", regex=True
    )

    # remove english
    idx = df["pred.content"].str.match(r"^[A-Za-z]")
    df.loc[idx, "pred.content"] = ""

    return df


def merge_aug(df0: pd.DataFrame, df_aug: pd.DataFrame) -> pd.DataFrame:
    # copy
    df = df0.copy()
    df_x = df_aug.copy()

    # check
    df.sample(1).iloc[0].to_dict()
    df_x.sample(1).iloc[0].to_dict()

    # gen merge_key
    df["merge_key"] = df["text.cc"] + "====" + df["text.zh"]
    df_x["merge_key"] = df_x["text.src"] + "====" + df_x["text.tgt"]

    # check
    assert df_x["merge_key"].is_unique
    (~df["merge_key"].isin(df_x["merge_key"])).sum()  # 0

    # add prefix to cols
    rcols = {c: f"_x.{c}" for c in df_x.columns if c != "merge_key"}
    df_x.rename(columns=rcols, inplace=True)

    # merge
    df = df.merge(df_x, on="merge_key", how="left")
    df.dropna(axis=1, how="all", inplace=True)
    df.sample(1).iloc[0].to_dict()

    # rename
    {k: k for k in df.columns}
    rcols = {
        "key": "key",
        "split": "split",
        "text.cc": "text.cc",
        "text.zh": "text.zh",
        "meta.data_id.cc": "meta.data_id.cc",
        "merge_key": "_merge_key",
        "_x.id": "_x.id",
        "_x.key": "_x.key",
        "_x.key2": "_x.key2",
        "_x.lang.src": "_x.lang.src",
        "_x.lang.tgt": "_x.lang.tgt",
        "_x.messages": "_x.messages",
        "_x.pred.completion_tokens": "_x.pred.completion_tokens",
        "_x.pred.content": "text.ko",
        "_x.pred.duration": "_x.pred.duration",
        "_x.pred.finish_reason": "_x.pred.finish_reason",
        "_x.pred.model": "meta.aug_model.ko",
        "_x.pred.model_dump_json": "_x.pred.model_dump_json",
        "_x.pred.model_dump_json_azure": "_x.pred.model_dump_json_azure",
        "_x.pred.prompt_tokens": "_x.pred.prompt_tokens",
        "_x.pred.service": "meta.aug_service.ko",
        "_x.split": "_x.split",
        "_x.text.src": "_x.text.src",
        "_x.text.tgt": "_x.text.tgt",
        "_x.meta.book_title.cc": "_x.meta.book_title.cc",
        "_x.meta.corpus": "_x.meta.corpus",
        "_x.meta.data_id.cc": "_x.meta.data_id.cc",
        "_x.meta.url.cc": "_x.meta.url.cc",
    }
    df.rename(columns=rcols, inplace=True)

    # drop
    dcols = [c for c in df.columns if c.startswith("_")]
    df.drop(columns=dcols, inplace=True)

    # sort cols
    c1 = [c for c in df.columns if not c.startswith("meta")]
    c2 = [c for c in df.columns if c.startswith("meta")]
    cols = sorted(c1) + sorted(c2)
    df = df[cols].reset_index(drop=True)
    df.sample(1).iloc[0].to_dict()

    # sort rows
    assert df["key"].is_unique
    df.sort_values(by="key", inplace=True)

    return df


def gen_align_file() -> None:
    # read
    df0 = utils.read_df(src.corpus.niu_mt.root.FILTER_PQ)
    df_aug = utils.read_df(src.eval.mt_aug.root.OUTPUT_GPT4_PQ)

    # fix
    df_aug = fix_aug(df_aug=df_aug)

    # check
    df0.sample(1).iloc[0].to_dict()
    df_aug.sample(1).iloc[0].to_dict()
    len(df0)  # 266514
    len(df_aug)  # 1115091

    # merge
    df = merge_aug(df0=df0, df_aug=df_aug)
    df.sample(1).iloc[0].to_dict()

    # check
    temp1 = df.isna().sum()
    temp1[temp1 > 0]
    df.sample(1).iloc[0].to_dict()

    # save
    utils.write_df2(sroot.ALIGN_PQ, df)


def main() -> None:
    # align samples
    gen_align_file()  # 153.4M, b631625f, 972467


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
            # python -m src.corpus.niu_mt_ko.align
            typer.run(main)
