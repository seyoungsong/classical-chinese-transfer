import random
import shutil
import sys
from importlib import reload
from pathlib import Path

import pandas as pd
import typer
from loguru import logger
from pandarallel import pandarallel
from rich import pretty
from tqdm import tqdm

import src.eval.mt_llm.root as sroot
from src import utils

LEFT_INDEX_STR = """
# HJ w/ CC
Unbabel/TowerInstruct-7B-v0.2--anonymous/TowerInstruct-7B-v0.2-CC-AWQ
anonymous/TowerInstruct-7B-v0.2-AJD-AWQ--anonymous/TowerInstruct-7B-v0.2-AJD-CC-AWQ
anonymous/TowerInstruct-7B-v0.2-AJD-KLC-AWQ--anonymous/TowerInstruct-7B-v0.2-AJD-KLC-CC-AWQ
# CC w/ HJ
anonymous/TowerInstruct-7B-v0.2-CC-AWQ--anonymous/TowerInstruct-7B-v0.2-AJD-CC-AWQ
anonymous/TowerInstruct-7B-v0.2-CC-AWQ--anonymous/TowerInstruct-7B-v0.2-AJD-KLC-CC-AWQ
"""


def gen_eval_table() -> None:
    # fix
    if 0:
        prev_dir = sroot.RESULT_DIR / "score"
        fnames = sorted(prev_dir.glob("*.json"))
        fnames = [p for p in fnames if str(p).count("--") == 2]
        f1 = random.choice(fnames)
        for f1 in tqdm(fnames):
            f2 = sroot.RESULT_DIR / "score_pair" / f1.name
            f2.parent.mkdir(exist_ok=True, parents=True)
            shutil.move(f1, f2)

    # read
    score_dir = sroot.RESULT_DIR / "score_pair"
    fnames = sorted(score_dir.glob("*.json"))
    df = pd.DataFrame([utils.read_json2(fname) for fname in fnames])
    df.groupby(["meta.corpus", "pred.model_id"]).size().value_counts()
    df.sort_values(["meta.corpus", "pred.model_id"], inplace=True, ignore_index=True)

    # check
    df.sample(1).iloc[0].to_dict()

    task = [
        ("bleu.score2.p_value", sroot.SCORE_BS_BLEU_TSV),
        ("spbleu.score2.p_value", sroot.SCORE_BS_SPBLEU_TSV),
        ("bleu.num_sample", sroot.SCORE_BS_NUM_SAMPLE_TSV),
    ]
    val_name: str
    fname1: Path
    if 0:
        val_name, fname1 = random.choice(task)
    for val_name, fname1 in task:

        # binary f1
        df1 = (
            df.groupby(["pred.model_id", "meta.corpus"])[val_name].mean()
        ).reset_index()
        df1 = df1.pivot(index="pred.model_id", columns="meta.corpus", values=val_name)
        df1 = df1.map(lambda x: f"{x:.6f}")

        # change order
        df.groupby(["meta.corpus", "pred.model_id"]).size()
        left_index = sorted(df1.index.unique())
        left_index = [
            x.strip()
            for x in LEFT_INDEX_STR.strip().split("\n")
            if x.strip() and "#" not in x
        ]
        print("\n".join(left_index))
        set(left_index) - set(df1.index.unique())
        set(df1.index.unique()) - set(left_index)

        #
        sorted(df1.columns.unique())
        top_columns = [
            "ajd-hj-en",
            "ajd-hj-ko",
            "drs-hj-ko",
            "drri-hj-ko",
            "klc-hj-ko",
            #
            "ocdb-cc-ko",
            # "cbc-cc-ko",
            #
            "niu-cc-ko",
            "niu-cc-zh",
            "wyweb_mt-cc-ko",
            "wyweb_mt-cc-zh",
        ]
        set(top_columns) - set(df1.columns.unique())
        set(df1.columns.unique()) - set(top_columns)

        # add empty rows if not exists
        for idx in left_index:
            if idx not in df1.index:
                df1.loc[idx] = ""

        # add empty columns if not exists
        for col in top_columns:
            if col not in df1.columns:
                df1[col] = ""

        df1 = df1.loc[left_index, top_columns]

        # save
        df1.fillna("", inplace=True)
        df1.replace("nan", "", inplace=True)
        df1.to_csv(fname1, sep="\t", index=True, encoding="utf-8-sig")
        utils.log_written(fname1)


def main() -> None:
    gen_eval_table()


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
            # python -m src.eval.mt_llm.eval_pair_table
            typer.run(main)
