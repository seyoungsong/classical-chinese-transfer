import sys
from importlib import reload
from typing import Any

import typer
from loguru import logger
from pandarallel import pandarallel
from rich import pretty
from tqdm import tqdm

import src.dataset.vocab.root as sroot
from src import utils


def merge_url(x1: dict[str, Any], rcols: dict[str, str]) -> str:
    urls = [(rcols[k], str(v)) for k, v in x1.items() if "url" in k and v is not None]
    urls = [(k, v) for k, v in urls if k != "ignore"]
    urls.sort(key=lambda x: x[0])
    urls_str = " | ".join([f"{k}: {v}" for k, v in urls])
    return urls_str


def gen_slim_file() -> None:
    # read
    df = utils.read_df(sroot.CONCAT_PQ)
    df.sample(1).iloc[0].to_dict()
    if 0:
        df = df.sample(frac=0.01, random_state=42).reset_index(drop=True)

    # merge url
    rcols = {c: c.replace("meta.", "") for c in df.columns if "url" in c}
    rcols = {
        "meta.data_url.cko": "ignore",
        "meta.data_url.en": "ignore",
        "meta.data_url.hj": "ignore",
        "meta.data_url.ko": "ignore",
        "meta.url.cc": "cc",
        "meta.url.cko": "cko",
        "meta.url.en": "en",
        "meta.url.hj": "hj",
        "meta.url.ko": "ko",
        "meta.url": "xx",
    }
    x1 = df.sample(1).iloc[0].to_dict()
    merge_url(x1=x1, rcols=rcols)
    df["url"] = df.progress_apply(lambda x1: merge_url(x1=x1, rcols=rcols), axis=1)  # type: ignore

    # drop meta cols
    dcols = [c for c in df.columns if "meta." in c or "text.type" == c]
    df.drop(columns=dcols, inplace=True)
    df.sample(1).iloc[0].to_dict()

    # sort cols
    c1 = [c for c in df.columns if not c.startswith("meta")]
    c2 = [c for c in df.columns if c.startswith("meta")]
    cols = sorted(c1) + sorted(c2)
    df = df[cols].reset_index(drop=True)
    df.sample(1).iloc[0].to_dict()

    # save
    utils.write_df2(sroot.SLIM_PQ, df)
    if 0:
        df = utils.read_df(sroot.SLIM_PQ)


def main() -> None:
    # remove meta columns
    gen_slim_file()  # 3.5G, 46275900, 5963484


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
            # python -m src.dataset.vocab.slim
            typer.run(main)
