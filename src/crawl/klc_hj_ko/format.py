import sys
from importlib import reload

import typer
from loguru import logger
from rich import pretty
from tqdm import tqdm

import src.crawl.klc_hj_ko.root as sroot
from src import utils


def gen_format_file() -> None:
    # init
    tqdm.pandas()

    # read
    df = utils.read_df(sroot.PARSE2_PQ)
    df.sample(1).iloc[0].to_dict()

    # rename cols
    {k: k for k in df.columns}
    rcols = {
        "meta.author": "meta.book_author",
        "meta.book_category": "meta.book_category",
        "meta.book_extra": "meta.book_extra_orig",
        "meta.book_extra_type": "meta.book_extra",
        "meta.book_id": "meta.book_id",
        "meta.book_title": "meta.book_title",
        "meta.data_id": "meta.data_id",
        "meta.elem_body_text": "text_xml",
        "meta.elem_col": "_meta.elem_col_idx",
        "meta.elem_copyright_text": "meta.elem_copyright",
        "meta.elem_dci_text": "meta.elem_dci",
        "meta.elem_id": "meta.elem_id",
        "meta.elem_id_type": "_meta.elem_id_type",
        "meta.elem_lang": "lang",
        "meta.elem_punc": "punc_type",
        "meta.elem_title_text": "meta.elem_title",
        "meta.elem_url": "meta.elem_url",
        "meta.mokcha_row": "_meta.mokcha_row",
        "meta.mokcha_title": "meta.data_title_mokcha",
        "meta.page_path": "meta.data_path",
        "meta.page_title": "meta.data_title",
        "meta.publisher": "meta.book_publisher",
        "meta.translator": "meta.book_translator",
        "meta.url": "meta.data_url",
        "meta.url2": "meta.url",
        "meta.year": "meta.book_year",
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

    if 0:
        x = df.sample(1).iloc[0].to_dict()
        print(x["text"])
        utils.open_code(x["meta.url"])


def main() -> None:
    # change format
    gen_format_file()  # 329.8M, 38de4bae


if __name__ == "__main__":
    if hasattr(sys, "ps1"):
        pretty.install()
        reload(utils)
        reload(sroot)
    else:
        with logger.catch(onerror=lambda _: sys.exit(1)):
            # python -m src.crawl.klc_hj_ko.format
            typer.run(main)
