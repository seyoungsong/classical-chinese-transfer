import sys
from importlib import reload

import typer
from loguru import logger
from rich import pretty
from tqdm import tqdm

import src.crawl.ajd_hj_oko.root as sroot
from src import utils


def gen_format_file() -> None:
    # init
    tqdm.pandas()

    # read
    df = utils.read_df(sroot.PARSE3_PQ)
    df.sample(1).iloc[0].to_dict()

    # rename cols
    {k: k for k in df.columns}
    rcols = {
        "meta.book_id": "meta.book_id",
        "meta.book_title": "meta.book_title",
        "meta.book_year": "meta.book_year",
        "meta.data_id": "meta.data_id",
        "meta.elem_body_text": "text_xml",
        "meta.elem_col": "_meta.elem_col",
        "meta.elem_id_type": "_meta.elem_id_type",
        "meta.elem_id": "meta.elem_id",
        "meta.elem_lang": "lang",
        "meta.elem_url": "meta.elem_url",
        "meta.mokcha_row": "_meta.mokcha_row",
        "meta.mokcha_title": "meta.data_title_mokcha",
        "meta.page_date": "meta.data_date",
        "meta.page_path": "meta.data_path",
        "meta.page_title": "meta.data_title",
        "meta.url": "meta.url",
    }
    df.rename(columns=rcols, inplace=True)

    # drop
    dcols = [c for c in df.columns if c.startswith("_")]
    df.drop(columns=dcols, inplace=True)

    # sort
    c1 = [c for c in df.columns if not c.startswith("meta")]
    c2 = [c for c in df.columns if c.startswith("meta")]
    cols = sorted(c1) + sorted(c2)
    df = df[cols].reset_index(drop=True)

    # write
    df.sample(1).iloc[0].to_dict()
    utils.write_df2(sroot.FORMAT_PQ, df)
    logger.debug(len(df))


def main() -> None:
    # rename and sort cols
    gen_format_file()  # 310.5M, a129eb3e, 826646


if __name__ == "__main__":
    if hasattr(sys, "ps1"):
        pretty.install()
        reload(utils)
        reload(sroot)
    else:
        with logger.catch(onerror=lambda _: sys.exit(1)):
            # python -m src.crawl.ajd_hj_oko.format
            typer.run(main)
