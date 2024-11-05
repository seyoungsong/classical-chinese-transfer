import sys
from importlib import reload

import humanize
import typer
from loguru import logger
from rich import pretty
from tqdm import tqdm

import src.corpus.wyweb_punc.root as sroot
from src import utils


def gen_stat_json() -> None:
    # init
    tqdm.pandas()

    # read
    df = utils.read_df(sroot.FILTER_PQ)
    df.sample(1).iloc[0].to_dict()

    # sample
    if 0:
        df3 = df[df["meta.data_id.ko"].notna()].reset_index(drop=True)
        df3.isna().sum()[df3.isna().sum() >= 1]
        utils.write_df2("sroot.MT_PQ", df3)

    # ratio_mt
    if 0:
        df.sample(1).iloc[0].to_dict()
        df["meta.data_id.ko"].notnull().mean() * 100  # 14.2
        c1 = df["text_body.ko"].notnull().sum() + df["text_title.ko"].notnull().sum()
        c2 = df["text_body.hj"].notnull().sum() + df["text_title.hj"].notnull().sum()
        c1 / c2 * 100  # 14.41

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
    if hasattr(sys, "ps1"):
        pretty.install()
        reload(utils)
        reload(sroot)
    else:
        with logger.catch(onerror=lambda _: sys.exit(1)):
            # python -m src.corpus.wyweb_punc.check
            typer.run(main)
