import sys
from importlib import reload
from pathlib import Path

import pandas as pd
import typer
from loguru import logger
from pandarallel import pandarallel
from rich import pretty
from tqdm import tqdm

import src.eval.ner1.root as sroot
from src import utils


def gen_eval_table() -> None:
    # read
    score_dir = sroot.RESULT_DIR / "score"
    fnames = sorted(score_dir.glob("*.json"))
    df = pd.DataFrame([utils.read_json2(fname) for fname in fnames])
    df.groupby(["meta.corpus", "pred.model_id"]).size().value_counts()
    df.sort_values(["meta.corpus", "pred.model_id"], inplace=True, ignore_index=True)

    # check
    df.sample(1).iloc[0].to_dict()

    task = [("binary.f1", sroot.F1_BINARY_TSV), ("entity.f1", sroot.F1_ENTITY_TSV)]
    val_name: str
    fname1: Path
    for val_name, fname1 in task:

        # binary f1
        df1 = (
            df.groupby(["pred.model_id", "meta.corpus"])[val_name].mean()
        ).reset_index()
        df1 = df1.pivot(index="pred.model_id", columns="meta.corpus", values=val_name)
        df1 = df1.map(lambda x: f"{x * 100:.4f}")

        # change order
        sorted(df1.index.unique())
        left_index = ["cc", "ajd", "ajd_cc", "ajd_klc", "ajd_klc_cc"]
        set(left_index) - set(df1.index.unique())
        set(df1.index.unique()) - set(left_index)

        #
        sorted(df1.columns.unique())
        top_columns = ["ajd", "klc", "histred_ner", "wyweb_ner"]
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
            # python -m src.eval.ner1.eval_table
            typer.run(main)
