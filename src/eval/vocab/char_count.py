import random
import sys
import unicodedata
from collections import Counter
from importlib import reload

import pandas as pd
import typer
from loguru import logger
from pandarallel import pandarallel
from rich import pretty
from tqdm import tqdm

import src.eval.vocab.root as sroot
import src.tool.eval as etool
from src import utils


def count_char(df1: pd.DataFrame) -> pd.DataFrame:
    counter = Counter()  # type: ignore
    if 0:
        s = df1["text"].sample(1).iloc[0]
        n = 2
    for s in tqdm(df1["text"]):
        # unigram (= char)
        s1 = etool.remove_punc(s)
        counter.update(s1)
        # n-gram
        if 0:
            for n in [2]:
                for i in range(len(s1) - n + 1):
                    ngram = s1[i : i + n]
                    counter[ngram] += 1
    df2 = pd.DataFrame(counter.most_common(), columns=["char", "count"])
    del counter
    df2.sort_values(
        ["count", "char"], inplace=True, ignore_index=True, ascending=[False, True]
    )
    df2["meta.corpus"] = df1["meta.corpus"].iloc[0]
    df2["lang"] = df1["lang"].iloc[0]
    return df2


def gen_char_count_file() -> None:
    # read file
    df = utils.read_df(sroot.DATASET_PQ)
    df.sample(1).iloc[0].to_dict()
    if 0:
        df = df.sample(n=10000, random_state=42).sort_index().reset_index(drop=True)

    # per corpus and lang
    if 0:
        utils.reset_dir(sroot.CHAR_COUNT_DIR)
        k1, df1 = random.choice([a for a in df.groupby(["meta.corpus", "lang"])])
    for k1, df1 in df.groupby(["meta.corpus", "lang"]):
        df2 = count_char(df1=df1)
        stem = "__".join(map(str, k1))
        fname = sroot.CHAR_COUNT_DIR / f"{stem}.parquet"
        fname.parent.mkdir(exist_ok=True, parents=True)
        utils.write_df(fname, df2)
        del df2

    # read and concat
    fnames = sorted(list(sroot.CHAR_COUNT_DIR.glob("*.parquet")))
    df_list = [utils.read_df(fname) for fname in tqdm(fnames)]
    df3 = pd.concat(df_list, ignore_index=True)
    df3["ngram"] = df3["char"].progress_apply(len)
    df3.sort_values(
        ["meta.corpus", "lang", "count", "char"],
        inplace=True,
        ignore_index=True,
        ascending=[True, True, False, True],
    )

    # save
    sroot.CHAR_COUNT_PQ.parent.mkdir(exist_ok=True, parents=True)
    utils.write_df2(sroot.CHAR_COUNT_PQ, df3)


def gen_char_count2_file() -> None:
    # read
    df = utils.read_df(sroot.CHAR_COUNT_PQ)
    df.sample(1).iloc[0].to_dict()

    # remove ngram
    if "ngram" in df.columns and df["ngram"].nunique() == 1:
        df.drop(columns=["ngram"], inplace=True)

    # check
    df.groupby(["lang", "meta.corpus"]).size()

    # convert: hj-royal, hj-lit, cc
    df.value_counts("meta.corpus")
    idx = df["meta.corpus"] == "klc"
    df.loc[idx, "lang"] = "hj_lit"
    idx = df["meta.corpus"].isin(["ajd", "drs", "drri"])
    df.loc[idx, "lang"] = "hj_royal"

    # drop: zh
    idx = df["lang"] == "zh"
    df = df[~idx].reset_index(drop=True)

    # merge by lang
    df.drop(columns=["meta.corpus"], inplace=True)
    df = df.groupby(["lang", "char"]).agg({"count": "sum"}).reset_index()
    df.sort_values(
        by=["lang", "count", "char"],
        inplace=True,
        ignore_index=True,
        ascending=[True, False, True],
    )
    df.value_counts("lang")
    df.sample(1).iloc[0].to_dict()

    # filter out korean
    df["is_kr"] = df["char"].progress_apply(utils.is_korean_char)
    df["is_kr"].mean() * 100  # 2.3
    df[df["is_kr"]]["char"].nunique()  # 1100
    df = df[~df["is_kr"]].reset_index(drop=True)
    df.drop(columns=["is_kr"], inplace=True)

    # check not chinese
    df["type"] = df["char"].progress_apply(unicodedata.category)
    df["type"].value_counts()
    df1 = df.groupby("type").agg({"char": lambda x: "".join(sorted(set(x.to_list())))})
    df1 = df1.reset_index()
    utils.write_json(sroot.RESULT_DIR / "char_type.json", df1.to_dict(orient="records"))

    # filter out not chinese
    not_ch_type_str = "Cc, Cf, Cn, Ll, Lu, Mc, Mn, Nd, Nl, No, Sc, Sk, Sm, So"
    not_ch_type = sorted(set(not_ch_type_str.split(", ")))
    ", ".join(not_ch_type)
    exception_is_ch = "㈠㈡㈢㈣㈤㈥㈦㈧㈨㈩㊋㊌㊍㊎㊏㊣"
    df["is_ch"] = ~df["type"].isin(not_ch_type) | df["char"].isin(set(exception_is_ch))
    df["is_ch"].mean() * 100  # 0.0
    df = df[df["is_ch"]].reset_index(drop=True)
    df.drop(columns=["is_ch"], inplace=True)

    # save
    df.drop(columns=["type"], inplace=True)
    sroot.CHAR_COUNT2_PQ.parent.mkdir(exist_ok=True, parents=True)
    utils.write_df2(sroot.CHAR_COUNT2_PQ, df)


def main() -> None:
    gen_char_count_file()  # 446.1K, ab108f57, 126260
    gen_char_count2_file()  # 275.9K, 9f795047, 63809


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
            # python -m src.eval.vocab.char_count
            typer.run(main)
