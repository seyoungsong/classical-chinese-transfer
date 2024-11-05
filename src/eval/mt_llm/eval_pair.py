import itertools
import random
import sys
from importlib import reload

import pandas as pd
import typer
from loguru import logger
from pandarallel import pandarallel
from rich import pretty
from tqdm import tqdm

import src.eval.mt_llm.root as sroot
import src.tool.eval as etool
import src.tool.eval.metric as emetric
from src import utils


def report_score(df1: pd.DataFrame, df2: pd.DataFrame) -> None:
    # check unique
    assert df1["pred.model_id"].nunique() == 1
    assert df2["pred.model_id"].nunique() == 1
    assert df1["meta.corpus"].nunique() == 1
    assert df2["meta.corpus"].nunique() == 1

    # check same corpus
    assert df1["meta.corpus"].iloc[0] == df2["meta.corpus"].iloc[0]

    # filter common keys
    df1 = df1[df1["key2"].isin(df2["key2"])].reset_index(drop=True)
    df2 = df2[df2["key2"].isin(df1["key2"])].reset_index(drop=True)

    # align
    df1.sort_values("key2", inplace=True, ignore_index=True)
    df2.sort_values("key2", inplace=True, ignore_index=True)
    assert (df1["key2"] == df2["key2"]).all()

    # check same ref1
    assert (df1["text.tgt"] == df2["text.tgt"]).all()

    # sample
    df1.sample(n=1).iloc[0].to_dict()
    if 0:
        df1 = df1.sample(n=100, random_state=0).sort_index().reset_index(drop=True)
        df2 = df2[df2["key2"].isin(df1["key2"])].reset_index(drop=True)

    # load
    ref1: list[str] = df1["text.tgt"].to_list()
    hyp1: list[str] = df1["pred.content"].to_list()
    hyp1 = [s if s is not None else "" for s in hyp1]
    hyp2: list[str] = df2["pred.content"].to_list()
    hyp2 = [s if s is not None else "" for s in hyp2]
    lang: str = df1["lang.tgt"].iloc[0]
    if 0:
        utils.temp_diff("\n\n".join(hyp1), "\n\n".join(ref1))
        utils.temp_diff("\n\n".join(hyp1), "\n\n".join(hyp2))

    # compute
    d1 = emetric.compute_bleu_paired_bs(
        hyp1=hyp1, hyp2=hyp2, ref1=ref1, lang=lang, mode="bleu"
    )
    d2 = emetric.compute_bleu_paired_bs(
        hyp1=hyp1, hyp2=hyp2, ref1=ref1, lang=lang, mode="spbleu"
    )

    # merge
    d1 = {f"bleu.{k}": v for k, v in d1.items()}
    d2 = {f"spbleu.{k}": v for k, v in d2.items()}
    d1.update(d2)

    # metadata
    d1["meta.corpus"] = df1["meta.corpus"].iloc[0]
    d1["pred.model_id.1"] = str(df1["pred.model_id"].iloc[0])
    d1["pred.model_id.2"] = str(df2["pred.model_id"].iloc[0])
    d1["pred.model_id"] = f"{d1['pred.model_id.1']}--{d1['pred.model_id.2']}"

    # save
    k1 = [d1["meta.corpus"], d1["pred.model_id"]]
    stem = "--".join(k1).replace("/", "__")
    fname = sroot.RESULT_DIR / f"score_pair/{stem}.json"
    fname.parent.mkdir(parents=True, exist_ok=True)
    utils.write_json2(fname, d1)


def main() -> None:
    # read
    df = utils.read_df(sroot.OUTPUT2_PQ)
    df.sample(1).iloc[0].to_dict()

    if 0:
        sz = df.isna().sum()
        sz[sz > 0]
        df1 = df[df["pred.content"].isna()].reset_index(drop=True)
        df1.sample(1).iloc[0].to_dict()
        df["pred.error"].value_counts()

    # remove error rows (?)
    df = df[df["pred.content"].notna()].reset_index(drop=True)

    # merge lang.src, lang.tgt to corpus
    if "meta.corpus.orig" not in df.columns:
        df["meta.corpus.orig"] = df["meta.corpus"]
    df["meta.corpus"] = (
        df["meta.corpus.orig"] + "-" + df["lang.src"] + "-" + df["lang.tgt"]
    )

    # check
    df.groupby(["key2", "pred.model_id"]).size().value_counts()
    df.groupby(["meta.corpus", "pred.model_id"]).size().to_dict()

    # find pairs (w/ lzh, w/o lzh)
    model_list: list[str] = sorted(df["pred.model_id"].unique())
    model_pair_list = [
        [s.replace("-CC", ""), s]
        for s in model_list
        if "CC" in s and s.replace("-CC", "") in model_list
    ]
    # find pairs (w/ hj, w/o hj)
    model_pair_list2 = [
        [
            "anonymous/TowerInstruct-7B-v0.2-CC-AWQ",
            "anonymous/TowerInstruct-7B-v0.2-AJD-CC-AWQ",
        ],
        [
            "anonymous/TowerInstruct-7B-v0.2-CC-AWQ",
            "anonymous/TowerInstruct-7B-v0.2-AJD-KLC-CC-AWQ",
        ],
    ]
    model_pair_list = sorted(model_pair_list + model_pair_list2)
    task_list: list[list[str]] = [
        [c, m[0], m[1]]
        for c, m in itertools.product(
            sorted(df["meta.corpus"].unique()), model_pair_list
        )
    ]

    # sequential
    if 0:
        corpus1, model1, model2 = random.choice(task_list)
    for corpus1, model1, model2 in tqdm(task_list):
        df1 = df[
            (df["meta.corpus"] == corpus1) & (df["pred.model_id"] == model1)
        ].reset_index(drop=True)
        df2 = df[
            (df["meta.corpus"] == corpus1) & (df["pred.model_id"] == model2)
        ].reset_index(drop=True)
        report_score(df1=df1, df2=df2)


if __name__ == "__main__":
    tqdm.pandas()
    pandarallel.initialize(progress_bar=True)
    #
    if hasattr(sys, "ps1"):
        pretty.install()
        reload(utils)
        reload(sroot)
        reload(etool)
        reload(emetric)
    else:
        with logger.catch(onerror=lambda _: sys.exit(1)):
            # python -m src.eval.mt_llm.eval_pair
            typer.run(main)
