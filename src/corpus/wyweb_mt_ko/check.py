import sys
from importlib import reload

import humanize
import typer
from loguru import logger
from rich import pretty
from tqdm import tqdm

import src.corpus.wyweb_mt_ko.root as sroot
from src import utils


def gen_stat_json() -> None:
    # read
    df = utils.read_df(sroot.FORMAT_PQ)
    df.sample(1).iloc[0].to_dict()

    # check
    cols = [c for c in df.columns if c.startswith("text") and "xml" not in c]
    df1 = df[cols].stack(dropna=False).reset_index(drop=True)
    df1.isna().mean()
    df1.dropna(inplace=True)
    df2 = df1.apply(len)
    d1 = df2.agg(["count", "mean", "median", "sum"]).astype(int).to_dict()
    d1["sum_human"] = humanize.intword(d1["sum"], format="%.0f")  # type: ignore
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
            # python -m src.corpus.wyweb_mt_ko.check
            typer.run(main)
