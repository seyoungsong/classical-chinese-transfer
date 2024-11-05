import re
import sys
from importlib import reload

import regex
import typer
from loguru import logger
from pandarallel import pandarallel
from rich import pretty
from tqdm import tqdm

import src.corpus.ocdb_mt.root as sroot
from src import utils


def count_char_korean(text: str) -> int:
    # Extended Korean characters range in Unicode
    pattern = re.compile(
        r"[\u1100-\u11ff\u3130-\u318f\u3200-\u321e\u3260-\u327f\uffa0-\uffdc\uffe6\uAC00-\uD7A3]"
    )
    matches = pattern.findall(text)
    return len(matches)


def count_char_punc(text: str) -> int:
    pattern = regex.compile(r"[\p{P}\s]")
    matches = pattern.findall(text)
    return len(matches)


def count_char_korean_punc_num(text: str) -> int:
    pattern = regex.compile(
        r"[\p{P}\s\d\u1100-\u11ff\u3130-\u318f\u3200-\u321e\u3260-\u327f\uffa0-\uffdc\uffe6\uAC00-\uD7A3]"
    )
    matches = pattern.findall(text)
    return len(matches)


def gen_filter_file() -> None:
    # read
    df = utils.read_df(sroot.SPLIT_PQ)
    df.sample(1).iloc[0].to_dict()

    # blank
    df = utils.replace_blank_to_none(df)

    # filter: text.ko should have at least 95% Korean characters
    df["num_kor"] = df["text.ko"].parallel_apply(count_char_korean_punc_num)
    df["len_kor"] = df["text.ko"].str.len()
    df["ratio_kor"] = df["num_kor"] / df["len_kor"]
    df["ratio_kor"].describe()
    df["ratio_kor"].quantile([i / 100 for i in range(0, 30, 1)])
    idx = df["ratio_kor"] >= 0.95
    idx.mean() * 100  # 83.38
    df = df[idx].reset_index(drop=True)

    # check
    df.isna().sum()[df.isna().sum() > 0]

    # sort cols
    c1 = [c for c in df.columns if not c.startswith("meta")]
    c2 = [c for c in df.columns if c.startswith("meta")]
    cols = sorted(c1) + sorted(c2)
    df = df[cols].reset_index(drop=True)
    df.sample(1).iloc[0].to_dict()

    # save
    utils.write_df2(sroot.FILTER_PQ, df)


def main() -> None:
    # filter poorly aligned data
    gen_filter_file()  # 19.0M, 20f1488a, 19841


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
            # python -m src.corpus.ocdb_mt.filter
            typer.run(main)
