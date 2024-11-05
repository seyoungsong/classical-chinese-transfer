import sys
from importlib import reload

import typer
from loguru import logger
from pandarallel import pandarallel
from rich import pretty
from tqdm import tqdm

import src.corpus.klc_ner.root as sroot
from src import utils


def gen_filter_file() -> None:
    # read
    df0 = utils.read_df(sroot.FORMAT_PQ)
    df = df0.copy()
    df.sample(1).iloc[0].to_dict()

    # is_punc should be True
    df["is_punc.hj"].value_counts() / len(df) * 100  # 98.9
    if 0:
        idx = ~df["is_punc.hj"]
        df[idx].sample(1).iloc[0].to_dict()  # 오주연문장전산고 등
    df = df[df["is_punc.hj"]].reset_index(drop=True)

    # check meta.data_copyright.hj
    df["meta.data_copyright.hj"].value_counts() / len(df) * 100  # 100

    # number of NE should be at least 2
    df["num_tags.hj"] = df["text_xml.hj"].str.count(f"</{utils.NER_PREF}")
    df["num_tags.hj"].value_counts().sort_index()
    idx = df["num_tags.hj"] >= 2
    idx.sum() / len(df) * 100  # 99.9
    df = df[idx].reset_index(drop=True)
    df.drop(columns=["num_tags.hj"], inplace=True)

    # check split
    logger.debug(df["split"].value_counts().sort_index())
    """
test      1035
test2     1154
train     8036
train2    9297
valid      995
valid2    1140
    """

    # sort rows
    df.sort_values("key", inplace=True, ignore_index=True)

    # sort cols
    c1 = [c for c in df.columns if not c.startswith("meta")]
    c2 = [c for c in df.columns if c.startswith("meta")]
    cols = sorted(c1) + sorted(c2)
    df = df[cols].reset_index(drop=True)
    df.sample(1).iloc[0].to_dict()

    # save
    utils.write_df2(sroot.FILTER_PQ, df)


def main() -> None:
    # rename columns, add key, sort rows, sort cols
    gen_filter_file()  # 64.2M, 1c4d6d56, 21657


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
            # python -m src.corpus.klc_ner.filter
            typer.run(main)
