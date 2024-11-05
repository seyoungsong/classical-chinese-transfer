import sys
from importlib import reload

import typer
from loguru import logger
from rich import pretty
from tqdm import tqdm

import src.dataset.ner1.root as sroot
from src import utils


def gen_stat_json() -> None:
    # read
    df = utils.read_df(sroot.FORMAT_PQ)
    df.sample(1).iloc[0].to_dict()

    # prepare
    d1 = {}
    df["len"] = df["text_xml"].str.len()

    # check
    d1["counts_per_split"] = df.groupby(["split"]).size().to_dict()
    cols = ["meta.corpus"]
    d1["counts_per_corpus"] = df.groupby(cols).size().to_dict()
    d1["total_len_per_corpus"] = df.groupby(cols)["len"].sum().to_dict()
    df.drop(columns=["len"], inplace=True)
    d1["sample_record"] = df.sample(1, random_state=42).iloc[0].to_dict()

    # convert
    d1 = utils.transform_keys(d1)
    sroot.STAT_JSON.parent.mkdir(parents=True, exist_ok=True)
    utils.write_json(sroot.STAT_JSON, d1)


def main() -> None:
    # check basic stats
    gen_stat_json()


if __name__ == "__main__":
    tqdm.pandas()
    #
    if hasattr(sys, "ps1"):
        pretty.install()
        reload(utils)
        reload(sroot)
    else:
        with logger.catch(onerror=lambda _: sys.exit(1)):
            # python -m src.dataset.ner1.check
            typer.run(main)
