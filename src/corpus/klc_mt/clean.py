import random
import sys
from importlib import reload

import typer
from loguru import logger
from pandarallel import pandarallel
from rich import pretty
from tqdm import tqdm

import src.corpus.klc_mt.root as sroot
from src import utils


def gen_clean_file() -> None:
    # read
    df0 = utils.read_df(sroot.FORMAT_PQ)
    df = df0.copy()

    # check
    if 0:
        df1 = df.dropna()
        x1 = df1.sample(1).iloc[0].to_dict()
        utils.write_json(utils.TEMP_JSON, x1)
        utils.open_code(utils.TEMP_JSON)

    # check title
    idx = df["meta.data_title.ko"] != df["meta.data_title2.ko"]
    idx.mean() * 100
    df[idx].sample(1).iloc[0].to_dict()
    idx = df["meta.data_title.ko"].str.len() >= df["meta.data_title2.ko"].str.len()
    idx.all()
    df[~idx].sample(1).iloc[0].to_dict()

    # 주-D004

    # text
    cols = [c for c in df.columns if c.startswith("text") and "xml" not in c]
    df2 = df[cols].reset_index(drop=True)
    if 0:
        x1 = df2.sample(1).iloc[0].to_dict()
        utils.write_json(utils.TEMP_JSON, x1)
        utils.open_code(utils.TEMP_JSON)

    # check full
    c = random.choice(df2.columns)
    d1 = {}
    for c in tqdm(df2.columns):
        vc = df2[c].value_counts()
        vc = vc[vc > 10]
        d1[c] = vc.to_dict()
    utils.write_json(utils.TEMP_JSON, d1)
    utils.open_code(utils.TEMP_JSON)

    # text.ko: 본문 내용 없음
    df["text.ko"].fillna("", inplace=True)
    idx = df["text.ko"].str.contains("본문 내용 없음", regex=False)
    idx.sum()  # 33
    df[idx].sample(1).iloc[0].to_dict()
    df.loc[idx, "text.ko"].str.len().value_counts()
    #
    idx1 = df["text.ko"].str.contains("본문 내용 없음", regex=False)
    idx2 = df["text.ko"].str.len() < 20
    idx = idx1 & idx2
    idx.sum()  # 29
    df[idx].sample(1).iloc[0].to_dict()
    cols = [c for c in df.columns if c.startswith("text")]
    df.loc[idx, cols] = ""

    # text.hj: 번역문 및 원문이미지 보기를 참고
    df["text.hj"].fillna("", inplace=True)
    idx = df["text.hj"].str.contains("원문이미지 보기를 참고", regex=False)
    idx.sum()  # 1
    df[idx].sample(1).iloc[0].to_dict()
    df.loc[idx, "text.hj"].str.len().value_counts()
    cols = [c for c in df.columns if c.startswith("text")]
    df.loc[idx, cols] = ""

    # check start
    c = random.choice(df2.columns)
    d1 = {}
    for c in tqdm(df2.columns):
        vc = df2[c].str[:5].value_counts()
        vc = vc[vc > 10]
        d1[c] = vc.to_dict()
    utils.write_json(utils.TEMP_JSON, d1)
    utils.open_code(utils.TEMP_JSON)

    # check end
    c = random.choice(df2.columns)
    d1 = {}
    for c in tqdm(df2.columns):
        vc = df2[c].str[-10:].value_counts()
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
    gen_clean_file()  # 618.9M, 173d953d, 157202


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
            # python -m src.corpus.klc_mt.clean
            typer.run(main)
