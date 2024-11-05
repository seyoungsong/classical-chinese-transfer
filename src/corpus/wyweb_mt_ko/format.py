import sys
from importlib import reload

import typer
from loguru import logger
from pandarallel import pandarallel
from rich import pretty
from tqdm import tqdm

import src.corpus.wyweb_mt_ko.root as sroot
from src import utils


def gen_format_file() -> None:
    # read
    df0 = utils.read_df(sroot.ALIGN_PQ)
    df = df0.copy()
    df.sample(1).iloc[0].to_dict()

    # fix model
    df["meta.aug_model.ko"].value_counts()
    {k: k for k in df["meta.aug_model.ko"].unique()}
    rvals = {"gpt-35-turbo": "gpt-3.5-turbo-0125", "gpt-4": "gpt-4-0125-preview"}
    df["meta.aug_model.ko"] = df["meta.aug_model.ko"].replace(rvals)

    # check service
    df["meta.aug_service.ko"].value_counts()

    # replace None
    df = utils.replace_blank_to_none(df)
    df.isna().sum()[df.isna().sum() > 0]

    # sort rows
    df.sort_values("key", inplace=True, ignore_index=True)

    # sort cols
    c1 = [c for c in df.columns if not c.startswith("meta")]
    c2 = [c for c in df.columns if c.startswith("meta")]
    cols = sorted(c1) + sorted(c2)
    df = df[cols].reset_index(drop=True)
    df.sample(1).iloc[0].to_dict()

    # save
    utils.write_df2(sroot.FORMAT_PQ, df)


def main() -> None:
    # rename columns, add key, sort rows, sort cols
    gen_format_file()  # 48.9M, 5881eef4, 266514


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
            # python -m src.corpus.wyweb_mt_ko.format
            typer.run(main)
