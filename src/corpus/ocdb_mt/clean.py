import random
import sys
from importlib import reload

import typer
from loguru import logger
from pandarallel import pandarallel
from rich import pretty
from tqdm import tqdm

import src.corpus.ocdb_mt.root as sroot
from src import utils


def gen_clean_file() -> None:
    # read
    df = utils.read_df(sroot.FORMAT_PQ)
    df.sample(1).iloc[0].to_dict()

    # text
    cols = [c for c in df.columns if c.startswith("text") and "xml" not in c]

    # check full
    c = random.choice(cols)
    d1 = {}
    for c in tqdm(cols):
        vc = df[c].value_counts()
        vc = vc[vc > 1]
        d1[c] = vc.to_dict()
    utils.write_json(utils.TEMP_JSON, d1)
    utils.open_code(utils.TEMP_JSON)

    # check start
    c = random.choice(cols)
    d1 = {}
    for c in tqdm(cols):
        vc = df[c].str[:5].value_counts()
        vc = vc[vc > 10]
        d1[c] = vc.to_dict()
    utils.write_json(utils.TEMP_JSON, d1)
    utils.open_code(utils.TEMP_JSON)

    # check
    if 0:
        idx = df["text.ko"].str.contains("Jam", regex=False)
        idx.sum()
        idx.mean()
        df[idx].sample(1).iloc[0].to_dict()

    # check
    if 0:
        s1 = "\n".join(df["text.cc"].sample(10))
        s2 = "\n".join(df["text.ko"].sample(10))
        s12 = s1 + "\n\n" + s2
        utils.write_str(utils.TEMP_TXT, s12)
        utils.open_code(utils.TEMP_TXT)

    # fix: 23-12. asdf
    pattern = r"^[\d\-\.\s]+"
    c = random.choice(cols)
    for c in cols:
        df[c] = df[c].str.replace(pattern, "", regex=True)

    # fix: [經] asdf, 【보】 asdf
    pattern = r"^(\[經\]|【보】|【綱】|\[강\])\s*"
    c = random.choice(cols)
    for c in cols:
        df[c] = df[c].str.replace(pattern, "", regex=True)

    # drop: [언해], [James Legge]
    idx1 = df["text.ko"].str.contains("[James Legge]", regex=False)
    idx2 = df["text.cc"].str.contains("[언해]", regex=False)
    idx = idx1 | idx2
    idx.mean() * 100
    df[idx].sample(1).iloc[0].to_dict()
    df = df[~idx].reset_index(drop=True)

    # check end
    c = random.choice(cols)
    d1 = {}
    for c in tqdm(cols):
        vc = df[c].str[-4:].value_counts()
        vc = vc[vc > 10]
        d1[c] = vc.to_dict()
    utils.write_json(utils.TEMP_JSON, d1)
    utils.open_code(utils.TEMP_JSON)

    # sort cols
    c1 = [c for c in df.columns if not c.startswith("meta")]
    c2 = [c for c in df.columns if c.startswith("meta")]
    cols = sorted(c1) + sorted(c2)
    df = df[cols].reset_index(drop=True)
    df.sample(1).iloc[0].to_dict()

    # save
    utils.write_df2(sroot.CLEAN_PQ, df)


def main() -> None:
    # clean texts if needed
    gen_clean_file()  # 22.8M, bc5dd4c0, 23795


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
            # python -m src.corpus.ocdb_mt.clean
            typer.run(main)
