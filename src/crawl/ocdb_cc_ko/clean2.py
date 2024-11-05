import sys
from importlib import reload

import typer
from loguru import logger
from pandarallel import pandarallel
from rich import pretty
from tqdm import tqdm

import src.crawl.ocdb_cc_ko.root as sroot
from src import utils


def gen_clean2_file() -> None:
    # read
    df = utils.read_df(sroot.CLEAN_PQ)
    df.sample(1).iloc[0].to_dict()

    print(", ".join(df.columns))
    # body.html, title.html, title.text, meta.bookname, meta.data_id, meta.data_url, meta.url

    # check: nan
    df.replace(r"^\s*$", None, regex=True, inplace=True)
    df.info()
    df.isna().sum()[df.isna().sum() > 0]

    # sample
    df.sample(1).iloc[0].to_dict()

    # save
    utils.write_df2(sroot.CLEAN2_PQ, df)


def main() -> None:
    # init
    tqdm.pandas()
    pandarallel.initialize(progress_bar=True)
    #
    gen_clean2_file()  # 60.3M, a621e3a9, 28341


if __name__ == "__main__":
    if hasattr(sys, "ps1"):
        pretty.install()
        reload(utils)
        reload(sroot)
        #
        tqdm.pandas()
        pandarallel.initialize(progress_bar=True)
    else:
        with logger.catch(onerror=lambda _: sys.exit(1)):
            # python -m src.crawl.ocdb_cc_ko.clean2
            typer.run(main)
