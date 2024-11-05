import sys
from importlib import reload

import typer
from loguru import logger
from pandarallel import pandarallel
from rich import pretty
from tqdm import tqdm

import src.crawl.drri_hj.root as sroot
from src import utils


def gen_clean2_file() -> None:
    # read
    df = utils.read_df(sroot.CLEAN_PQ)

    print(", ".join(df.columns))
    # meta.book_id, meta.book_id_vol, meta.data_url, meta.elem_body_html, meta.elem_body_text, meta.elem_idx_lv1, meta.elem_idx_lv2, meta.elem_title_html, meta.elem_title_text, meta.page_date, meta.url
    df.sample(1).iloc[0].to_dict()

    # empty to nan
    df.replace(r"^\s*$", None, regex=True, inplace=True)
    df.info()
    df.isna().sum()[df.isna().sum() > 0]

    # ok to be empty for now
    df.fillna("", inplace=True)

    # sample
    df.sample(1).iloc[0].to_dict()

    # merge
    df["meta.book_id"] = df["meta.book_id"] + df["meta.book_id_vol"].apply(
        lambda x: f"_{x:04d}"
    )
    df.drop(columns=["meta.book_id_vol"], inplace=True)

    # sort rows
    kcols = ["meta.data_url", "meta.elem_idx_lv1", "meta.elem_idx_lv2"]
    df.groupby(kcols).size().value_counts()
    df.sort_values(kcols, inplace=True, ignore_index=True)

    # add data_id (custom index)
    df["meta.data_id"] = df.index + 1
    digit = len(str(df["meta.data_id"].max()))
    df["meta.data_id"] = df["meta.data_id"].apply(lambda x: f"L{x:0{digit}d}")

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
    gen_clean2_file()  # 164.9M, 441bc6c2


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
            # python -m src.crawl.drri_hj.clean2
            typer.run(main)
