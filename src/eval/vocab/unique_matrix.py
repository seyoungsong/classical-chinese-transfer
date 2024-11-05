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

import src.eval.vocab.root as sroot
from src import utils


def gen_unique_matrix_json() -> None:
    # read
    df = utils.read_df(sroot.CHAR_COUNT2_PQ)
    df.sample(1).iloc[0].to_dict()

    # by lang
    langs: list[str] = df["lang"].unique().tolist()
    lang_pairs: list[list[str]] = list(itertools.product(langs, repeat=2))  # type: ignore
    lang_pairs = [pair for pair in lang_pairs if pair[0] != pair[1]]

    lang1, lang2 = random.choice(lang_pairs)
    threshold = 99.9
    ratio_list = []
    for lang1, lang2 in lang_pairs:
        for threshold in [100.0, 99.99, 99.9, 99.0, 95.0, 90.0]:
            df1 = df[df["lang"] == lang1].reset_index(drop=True)
            df2 = df[df["lang"] == lang2].reset_index(drop=True)
            df2.sort_values("count", inplace=True, ignore_index=True, ascending=False)
            df2["cumpct"] = df2["count"].cumsum() / df2["count"].sum() * 100
            df2 = df2[df2["cumpct"] <= threshold]
            idx = df1["char"].isin(df2["char"])
            ratio = df1[~idx]["count"].sum() / df1["count"].sum()
            ratio_list.append(
                {"lang1": lang1, "lang2": lang2, "ratio": ratio, "threshold": threshold}
            )
    df_ratio = pd.DataFrame(ratio_list)
    df_ratio.sort_values(
        ["lang1", "lang2", "threshold", "ratio"], inplace=True, ignore_index=True
    )
    utils.write_df(sroot.RESULT_DIR / "unique_matrix.json", df_ratio)


def gen_tsv() -> None:
    # read
    df = utils.read_df(sroot.RESULT_DIR / "unique_matrix.json")
    df.sample(1).iloc[0].to_dict()

    # convert
    df["threshold"] = df["threshold"].apply(lambda x: f"{x:.2f}p")
    df["ratio"] = df["ratio"].apply(lambda x: f"{x:.4%}")

    if 0:
        df1 = random.choice([df1 for _, df1 in df.groupby("threshold")])
    for _, df1 in df.groupby("threshold"):
        df2 = df1.pivot(index="lang1", columns="lang2", values="ratio").fillna("-")
        col_order = ["hj_royal", "hj_lit", "cc"]
        df2 = df2[col_order]
        idx_order = ["hj_royal", "hj_lit", "cc"]
        df2 = df2.loc[idx_order]
        threshold1 = df1["threshold"].iloc[0]
        fname1 = sroot.RESULT_DIR / f"unique_matrix/threshold_{threshold1}.tsv"
        fname1.parent.mkdir(parents=True, exist_ok=True)
        df2.to_csv(fname1, sep="\t", index=True, header=True)
        utils.log_written(fname1)


def main() -> None:
    gen_unique_matrix_json()  # 2.3K, 74321791, 24
    gen_tsv()


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
            # python -m src.eval.vocab.unique_matrix
            typer.run(main)
