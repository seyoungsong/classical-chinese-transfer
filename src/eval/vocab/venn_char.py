import sys
from importlib import reload

import typer
from loguru import logger
from pandarallel import pandarallel
from rich import pretty
from tqdm import tqdm

import src.eval.vocab.root as sroot
from src import utils


def gen_venn_char_tsv() -> None:
    # read
    df = utils.read_df(sroot.CHAR_COUNT2_PQ)
    df.sample(1).iloc[0].to_dict()

    # unique char per corpus combined
    df1 = (
        df.groupby("char")
        .agg({"lang": lambda x: "|".join(x), "count": "sum"})
        .reset_index()
    )
    df1.sort_values(
        by=["lang", "count", "char"],
        inplace=True,
        ignore_index=True,
        ascending=[True, False, True],
    )

    # check
    df1["lang"].value_counts()

    # save
    fname = sroot.MODEL_DIR / "venn_char.tsv"
    fname.parent.mkdir(exist_ok=True, parents=True)
    utils.write_df(fname, df1)

    # save
    df2 = df1["lang"].value_counts().reset_index().sort_values("lang")
    df2["lang"] = df2["lang"].str.replace("|", " ∩ ")
    df2["lang"] = df2["lang"].str.replace("cc", "Lzh")
    df2["lang"] = df2["lang"].str.replace("hj_lit", "Hj (L)")
    df2["lang"] = df2["lang"].str.replace("hj_royal", "Hj (R)")
    df2.sort_values("lang", inplace=True, ignore_index=True)
    fname = sroot.RESULT_DIR / "venn_char_size.tsv"
    fname.parent.mkdir(exist_ok=True, parents=True)
    utils.write_df(fname, df2)


def gen_venn_char2_tsv() -> None:
    # read
    df = utils.read_df(sroot.CHAR_COUNT2_PQ)
    df.sample(1).iloc[0].to_dict()

    # convert: hj-rl, cc
    df.value_counts("lang")
    idx = df["lang"] == "hj_lit"
    df.loc[idx, "lang"] = "hj_rl"
    idx = df["lang"].isin(["hj_royal"])
    df.loc[idx, "lang"] = "hj_rl"

    # merge by lang
    df = df.groupby(["lang", "char"]).agg({"count": "sum"}).reset_index()
    df.sort_values(
        by=["lang", "count", "char"],
        inplace=True,
        ignore_index=True,
        ascending=[True, False, True],
    )
    df.value_counts("lang")
    df.sample(1).iloc[0].to_dict()

    # unique char per corpus combined
    df1 = (
        df.groupby("char")
        .agg({"lang": lambda x: "|".join(x), "count": "sum"})
        .reset_index()
    )
    df1.sort_values(
        by=["lang", "count", "char"],
        inplace=True,
        ignore_index=True,
        ascending=[True, False, True],
    )

    # check
    df1["lang"].value_counts()

    # save
    fname = sroot.MODEL_DIR / "venn_char2.tsv"
    fname.parent.mkdir(exist_ok=True, parents=True)
    utils.write_df(fname, df1)

    # save
    df2 = df1["lang"].value_counts().reset_index().sort_values("lang")
    df2["lang"] = df2["lang"].str.replace("|", " ∩ ")
    df2["lang"] = df2["lang"].str.replace("cc", "Lzh")
    df2["lang"] = df2["lang"].str.replace("hj_rl", "Hj")
    df2.sort_values("lang", inplace=True, ignore_index=True)
    fname = sroot.RESULT_DIR / "venn_char2_size.tsv"
    fname.parent.mkdir(exist_ok=True, parents=True)
    utils.write_df(fname, df2)


def gen_venn_char_sample_tsv() -> None:
    # 1
    fname = sroot.MODEL_DIR / "venn_char.tsv"
    df1 = utils.read_df(fname)
    df1.sort_values("count", inplace=True, ignore_index=True, ascending=False)
    df1["lang"].value_counts()
    df2 = df1.groupby("lang").agg({"char": lambda x: " ".join(x[:100])}).reset_index()
    df2["lang"] = df2["lang"].str.replace("|", " ∩ ")
    df2["lang"] = df2["lang"].str.replace("cc", "Lzh")
    df2["lang"] = df2["lang"].str.replace("hj_lit", "Hj (L)")
    df2["lang"] = df2["lang"].str.replace("hj_royal", "Hj (R)")
    df2.sort_values("lang", inplace=True, ignore_index=True)
    utils.write_df(sroot.RESULT_DIR / "venn_char_sample.tsv", df2)

    # 2
    fname = sroot.MODEL_DIR / "venn_char2.tsv"
    df1 = utils.read_df(fname)
    df1.sort_values("count", inplace=True, ignore_index=True, ascending=False)
    df1["lang"].value_counts()
    df2 = df1.groupby("lang").agg({"char": lambda x: " ".join(x[:100])}).reset_index()
    df2["lang"] = df2["lang"].str.replace("|", " ∩ ")
    df2["lang"] = df2["lang"].str.replace("cc", "Lzh")
    df2["lang"] = df2["lang"].str.replace("hj_rl", "Hj")
    df2.sort_values("lang", inplace=True, ignore_index=True)
    utils.write_df(sroot.RESULT_DIR / "venn_char2_sample.tsv", df2)


def main() -> None:
    gen_venn_char_tsv()
    gen_venn_char2_tsv()
    gen_venn_char_sample_tsv()


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
            # python -m src.eval.vocab.venn_char
            typer.run(main)
