import sys
from importlib import reload

import typer
from loguru import logger
from pandarallel import pandarallel
from rich import pretty
from tqdm import tqdm

import src.corpus.klc_mt.root as sroot
from src import utils


def gen_format_file() -> None:
    # read
    df0 = utils.read_df(sroot.ALIGN_PQ)
    df = df0.copy()

    # check
    if 0:
        df1 = df.dropna()
        x1 = df1.sample(1).iloc[0].to_dict()
        utils.write_json(utils.TEMP_JSON, x1)
        utils.open_code(utils.TEMP_JSON)

    # drop temp
    cols = [c for c in df.columns if c.startswith("temp.")]
    df.drop(columns=cols, inplace=True)

    # "meta.elem_id.ko" -> "meta.data_id.ko"
    assert df["meta.data_id.ko"].equals(df["meta.elem_id.ko"])
    df.drop(columns=["meta.data_id.ko"], inplace=True)
    df.rename(columns={"meta.elem_id.ko": "meta.data_id.ko"}, inplace=True)
    df.rename(columns={"meta.elem_id.hj": "meta.data_id.hj"}, inplace=True)

    # "meta.book_id.ko" -> None
    assert (df["meta.data_id.ko"].str[:13] == df["meta.book_id.ko"]).all()
    df.drop(columns=["meta.book_id.ko"], inplace=True)

    # "meta.elem_url.ko" -> None
    df["meta.elem_url.ko"].str.len().value_counts()
    assert (df["meta.url.ko"].str[:85] == df["meta.elem_url.ko"]).all()
    df.drop(columns=["meta.elem_url.ko"], inplace=True)

    # rename ko -> hj, and others
    rcols = {
        "meta.book_author.ko": "meta.book_author.hj",
        "meta.book_category.ko": "meta.book_category.hj",
        "meta.book_extra.ko": "meta.book_extra.hj",
        "meta.book_extra_orig.ko": "meta.book_extra_orig.hj",
        "meta.book_publisher.ko": "meta.book_publisher.hj",
        "meta.book_title.ko": "meta.book_title.hj",
        "meta.book_year.ko": "meta.book_year.hj",
        "meta.data_title.ko": "meta.data_path2.ko",
        "meta.data_title_mokcha.ko": "meta.data_title2.ko",
        "meta.elem_url.hj": "meta.url.hj",
        "meta.elem_copyright.hj": "meta.data_copyright.hj",
        "meta.elem_copyright.ko": "meta.data_copyright.ko",
        "meta.elem_dci.hj": "meta.data_dci.hj",
        "meta.elem_dci.ko": "meta.data_dci.ko",
        "meta.elem_title.hj": "meta.data_title.hj",
        "meta.elem_title.ko": "meta.data_title.ko",
    }
    df.rename(columns=rcols, inplace=True)

    # check key
    if 0:
        nu = df.nunique()
        nu[nu == nu.max()]
    assert df["meta.data_id.ko"].is_unique, "key is not unique"
    df["key"] = df["meta.data_id.ko"]

    # sort rows
    df.sort_values("key", inplace=True, ignore_index=True)

    # replace None
    df = utils.replace_blank_to_none(df)

    # sort cols
    c1 = [c for c in df.columns if not c.startswith("meta")]
    c2 = [c for c in df.columns if c.startswith("meta")]
    cols = sorted(c1) + sorted(c2)
    df = df[cols].reset_index(drop=True)
    df.sample(1).iloc[0].to_dict()

    # check
    if 0:
        df1 = df.dropna()
        x1 = df1.sample(1).iloc[0].to_dict()
        utils.write_json(utils.TEMP_JSON, x1)
        utils.open_code(utils.TEMP_JSON)

    # save
    utils.write_df2(sroot.FORMAT_PQ, df)


def main() -> None:
    # rename columns, add key, sort rows, sort cols
    gen_format_file()  # 619.6M, ff10fbb3


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
            # python -m src.corpus.klc_mt.format
            typer.run(main)
