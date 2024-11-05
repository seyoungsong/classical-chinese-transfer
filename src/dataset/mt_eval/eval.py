import random
import sys
from importlib import reload
from pathlib import Path

import pandas as pd
import typer
from loguru import logger
from pandarallel import pandarallel
from rich import pretty
from tqdm import tqdm

import src.dataset.mt_eval.root as sroot
from src import utils

NUM_TEST = 6000  # alma-r is 2K, ko-en is 1965


def gen_eval_file() -> None:
    # read
    df = utils.read_df(sroot.FILTER4_PQ)
    df.sample(1).iloc[0].to_dict()

    # check: test only
    df["split"].value_counts()
    assert df["split"].eq("test").all(), 'split should be "test"'

    # read prev eval
    f_prev = Path(str(sroot.EVAL_PQ).replace("_v3", "_v1")).resolve()
    df_prev = utils.read_df(f_prev)
    df_prev["key2"].value_counts().value_counts()
    df_prev.groupby(["lang.src", "lang.tgt", "meta.corpus"]).size()

    # filter out unknown
    idx = df_prev["key2"].isin(df["key2"])
    idx.mean() * 100  # 99.84, not 100 due to fluctations in dedup step
    df_prev = df_prev[idx].reset_index(drop=True)

    # drop bad samples
    bad_str = f"<{utils.NER_PREF}"
    idx = df["text.src"].str.contains(bad_str) | df["text.tgt"].str.contains(bad_str)
    idx.sum()  # 5
    if 0:
        df[idx].sample(1).iloc[0].to_dict()
    df = df[~idx].reset_index(drop=True)

    # check
    size = df.groupby(["lang.src", "lang.tgt", "meta.corpus"]).size()
    fname = sroot.RESULT_DIR / "filter4_size.txt"
    fname.parent.mkdir(parents=True, exist_ok=True)
    utils.write_str(fname, size.to_string())

    # first, get all samples from prev eval by key2
    idx = df["key2"].isin(df_prev["key2"])
    idx.sum()
    df1_prev = df[idx].reset_index(drop=True)
    df_remain = df[~idx].reset_index(drop=True)
    df_remain.groupby(["lang.src", "lang.tgt", "meta.corpus"]).size()

    # random sample
    df1_list = [df1_prev]
    if 0:
        k1, df_k = random.choice(
            list(df_remain.groupby(["meta.corpus", "lang.src", "lang.tgt"]))
        )
    for k1, df_k in df_remain.groupby(["meta.corpus", "lang.src", "lang.tgt"]):
        idx1 = df1_prev["meta.corpus"] == str(k1[0])
        idx2 = df1_prev["lang.src"] == str(k1[1])
        idx3 = df1_prev["lang.tgt"] == str(k1[2])
        idx = idx1 & idx2 & idx3
        num_test_k = NUM_TEST - idx.sum()
        if num_test_k < 0:
            num_test_k = 0
        df_k = df_k.sample(
            n=min(num_test_k, len(df_k)), random_state=42, ignore_index=True
        )
        df1_list.append(df_k)

    # concat
    df1 = pd.concat(df1_list, ignore_index=True)
    assert df1["key2"].is_unique
    df1.sort_values("key2", inplace=True, ignore_index=True)
    df1.groupby(["lang.src", "lang.tgt", "meta.corpus"]).size()

    # save
    utils.write_df2(sroot.EVAL_PQ, df1)


def main() -> None:
    # select small test subset
    gen_eval_file()  # 19.0M, 553a7723, 57979


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
            # python -m src.dataset.mt_eval.eval
            typer.run(main)
