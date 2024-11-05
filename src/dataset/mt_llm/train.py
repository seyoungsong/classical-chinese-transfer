import sys
from importlib import reload

import pandas as pd
import typer
from loguru import logger
from pandarallel import pandarallel
from rich import pretty
from tqdm import tqdm

import src.dataset.mt_llm.root as sroot
from src import utils

NUM_SAMPLES = 10000


def gen_train_file() -> None:
    # read
    df0 = utils.read_df(sroot.FILTER_PQ)
    df = df0.copy()
    df.sample(1).iloc[0].to_dict()

    # filter: keep train only
    idx = df["split"] == "train"
    df = df[idx].reset_index(drop=True)

    # check: stat
    df.info()
    df.groupby(["meta.corpus", "lang"], dropna=False).size()
    if 1:
        temp1 = df.groupby(["meta.corpus", "lang"], dropna=False)["len"].agg(
            ["sum", "count"]
        )
        temp1.reset_index(inplace=True)
        temp1["sum"] = round(temp1["sum"] / temp1["sum"].sum() * 100, 1)
        logger.debug(temp1)

    """
  meta.corpus   lang   sum   count
0         ajd  hj-en   1.2   15384 -> 10K
1         ajd  hj-ko  30.8  262234 -> 10K
2         ajd  ko-en   1.5   14719 -> 10K
3         klc  hj-ko   6.4   37519 -> 10K
4         niu  cc-ko  28.0  722554 -> 5K
5         niu  cc-zh  20.6  723737 -> 5K (same id as cc-ko)
6    wyweb_mt  cc-ko   6.7  161081 -> 5K
7    wyweb_mt  cc-zh   4.8  161166 -> 5K (same id as cc-ko)
    """

    # filter: sample random per corpus and lang (to reduce train time)
    # note: niu and wyweb_mt should be half-sized.
    # note: cc-ko and cc-zh should be same id.
    # note: max size should be similar to ALMA paper. (~58k, about 10k-100k)

    # hj
    idx = df["lang"].str.contains("cc")
    df_hj = df[~idx].reset_index(drop=True)

    # cc
    df_ck = df[df["lang"] == "cc-ko"].reset_index(drop=True)
    df_cz = df[df["lang"] == "cc-zh"].reset_index(drop=True)
    idx = df_ck["key"].isin(df_cz["key"])
    common_key = df_ck[idx]["key"].reset_index(drop=True)
    df_ck = df_ck[df_ck["key"].isin(common_key)].reset_index(drop=True)
    df_cz = df_cz[df_cz["key"].isin(common_key)].reset_index(drop=True)
    assert df_ck["key"].equals(df_cz["key"])

    # hj-xx
    df_hj1 = (
        df_hj.groupby(["meta.corpus", "lang"], dropna=False)
        .apply(lambda x: x.sample(n=min(len(x), NUM_SAMPLES), random_state=42))
        .reset_index(drop=True)
    )
    df_hj1.groupby(["meta.corpus", "lang"], dropna=False).size()
    df_hj1.sort_values("key2", inplace=True, ignore_index=True)
    logger.debug(f"{len(df_hj1) / len(df):.1%}")

    # cc-ko
    df_ck1 = (
        df_ck.groupby(["meta.corpus", "lang"], dropna=False)
        .apply(lambda x: x.sample(n=min(len(x), NUM_SAMPLES // 2), random_state=42))
        .reset_index(drop=True)
    )
    df_ck1.groupby(["meta.corpus", "lang"], dropna=False).size()
    df_ck1.sort_values("key2", inplace=True, ignore_index=True)
    logger.debug(f"{len(df_ck1) / len(df):.1%}")

    # cc-zh
    idx = df_cz["key"].isin(df_ck1["key"])
    df_cz1 = df_cz[idx].reset_index(drop=True)
    df_cz1.sort_values("key2", inplace=True, ignore_index=True)
    assert df_cz1["key"].equals(df_ck1["key"]), "not same key"

    # concat
    df = pd.concat([df_hj1, df_ck1, df_cz1], ignore_index=True)
    df.sort_values("key2", inplace=True, ignore_index=True)
    if 1:
        temp1 = df.groupby(["meta.corpus", "lang"], dropna=False)["len"].agg(
            ["sum", "count"]
        )
        temp1.reset_index(inplace=True)
        temp1["sum"] = round(temp1["sum"] / temp1["sum"].sum() * 100, 1)
        logger.debug(temp1)

    """
  meta.corpus   lang   sum  count
0         ajd  hj-en  14.8  10000
1         ajd  hj-ko  21.6  10000
2         ajd  ko-en  19.0  10000
3         klc  hj-ko  31.4  10000
4         niu  cc-ko   3.6   5000
5         niu  cc-zh   2.7   5000
6    wyweb_mt  cc-ko   4.0   5000
7    wyweb_mt  cc-zh   2.9   5000
    """

    # save
    utils.write_df2(sroot.TRAIN_PQ, df)


def main() -> None:
    gen_train_file()  # 19.6M, db5c71aa, 60000


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
            # python -m src.dataset.mt_llm.train
            typer.run(main)
