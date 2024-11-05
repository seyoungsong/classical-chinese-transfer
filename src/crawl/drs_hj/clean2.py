import sys
from importlib import reload

import typer
from loguru import logger
from pandarallel import pandarallel
from rich import pretty
from tqdm import tqdm

import src.crawl.drs_hj.root as sroot
from src import utils


def gen_clean2_file() -> None:
    # read
    df = utils.read_df(sroot.CLEAN_PQ)

    print(", ".join(df.columns))
    # meta.book_id, meta.book_title, meta.book_year, meta.data_id, meta.elem_body_html, meta.elem_body_text, meta.elem_btn_ko_html, meta.elem_btn_orig_html, meta.elem_col, meta.elem_id, meta.elem_url, meta.mokcha_row, meta.mokcha_title, meta.page_date, meta.page_path, meta.page_title, meta.url
    df.sample(1).iloc[0].to_dict()

    # check trivial cols
    temp1: utils.SeriesType = df.progress_apply(  # type: ignore
        lambda x: x["meta.data_id"].startswith(x["meta.book_id"]), axis=1
    )
    assert temp1.all(), "meta.data_id should startswith meta.book_id"
    #
    assert (
        len(df["meta.elem_col"].value_counts()) == 1
    ), "meta.elem_col should be all the same"
    #
    assert (
        df["meta.elem_id"] == df["meta.data_id"]
    ).all(), "meta.elem_id should be the same as meta.data_id"

    # drop trivial cols
    dcols = [
        "meta.book_id",
        "meta.elem_col",
        "meta.elem_id",
        "meta.elem_url",
        "meta.mokcha_row",
        "meta.mokcha_title",
        "meta.page_path",
    ]
    df.drop(columns=dcols, inplace=True)

    # check & sort
    kcols = ["meta.data_id"]
    df.groupby(kcols).size().value_counts()
    df.sort_values(kcols, inplace=True, ignore_index=True)

    # empty to nan
    df.replace(r"^\s*$", None, regex=True, inplace=True)
    df.info()
    df.isna().sum()[df.isna().sum() > 0]

    # ok to be empty for now
    df.fillna("", inplace=True)

    # sample
    df.sample(1).iloc[0].to_dict()

    # sort rows
    kcols = ["meta.data_id"]
    df.groupby(kcols).size().value_counts()
    df.sort_values(kcols, inplace=True, ignore_index=True)

    # sort cols
    df = df[sorted(df.columns)].reset_index(drop=True)

    # save
    df.info()
    utils.write_df2(sroot.CLEAN2_PQ, df)


def main() -> None:
    # drop some rows, and add some columns
    # init
    tqdm.pandas()
    pandarallel.initialize(progress_bar=True)
    #
    gen_clean2_file()  # 735.7M, 29db5b19


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
            # python -m src.crawl.drs_hj.clean2
            typer.run(main)
