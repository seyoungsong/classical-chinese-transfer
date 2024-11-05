import sys
from importlib import reload

import typer
from loguru import logger
from pandarallel import pandarallel
from rich import pretty
from tqdm import tqdm

import src.corpus.klc_hj.root as sroot
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

    # check
    df["meta.book_extra.hj"].value_counts()
    df["meta.book_id.hj"].str.len().value_counts()
    assert (df["meta.data_id.hj"].str[:13] == df["meta.book_id.hj"]).all()
    assert (df["meta.elem_id.hj"] == df["meta.data_id.hj"]).all()

    # drop
    dcols = ["meta.book_id.hj", "meta.elem_id.hj", "meta.elem_url.hj"]
    df.drop(columns=dcols, inplace=True)

    # rename1
    rcols = {
        "meta.data_title.hj": "meta.data_path2.hj",
    }
    df.rename(columns=rcols, inplace=True)

    # rename2
    rcols = {
        "meta.elem_url.hj": "meta.url.hj",
        "meta.elem_copyright.hj": "meta.data_copyright.hj",
        "meta.elem_dci.hj": "meta.data_dci.hj",
        "meta.elem_title.hj": "meta.data_title.hj",
        "meta.data_title_mokcha.hj": "meta.data_title2.hj",
    }
    df.rename(columns=rcols, inplace=True)

    # check key
    if 0:
        nu = df.nunique()
        nu[nu == nu.max()]
    assert df["meta.data_id.hj"].is_unique, "key is not unique"
    df["key"] = df["meta.data_id.hj"]

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
            # python -m src.corpus.klc_hj.format
            typer.run(main)
