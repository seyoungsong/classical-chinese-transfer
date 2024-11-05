import sys
from importlib import reload

import typer
from loguru import logger
from pandarallel import pandarallel
from rich import pretty
from tqdm import tqdm

import src.corpus.drs.root as sroot
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
    df.drop(columns=["meta.data_id.ko"], inplace=True)
    df.rename(columns={"meta.elem_id.ko": "meta.data_id.ko"}, inplace=True)

    # "meta.elem_url.ko" -> "meta.url.ko"
    df.drop(columns=["meta.url.ko"], inplace=True)
    df.rename(columns={"meta.elem_url.ko": "meta.url.ko"}, inplace=True)

    # "meta.data_title.ko" -> "meta.data_date.ko"
    df.rename(columns={"meta.data_title.ko": "meta.data_date.ko"}, inplace=True)
    # "meta.elem_title.ko" -> "meta.data_title.ko"
    df.rename(columns={"meta.elem_title.ko": "meta.data_title.ko"}, inplace=True)

    # "meta.data_id_ko.hj" -> "meta.link_id.hj"
    df.rename(columns={"meta.data_id_ko.hj": "meta.link_id.hj"}, inplace=True)

    if 0:
        # "meta.book_id.ko" drop
        df1 = df[df["meta.book_id.ko"].notna()].reset_index(drop=True)
        idx = df1["meta.book_id.ko"] == df1["meta.data_id.ko"].str[:10]
        assert idx.all(), "bad meta.book_id.ko"
        df.drop(columns=["meta.book_id.ko"], inplace=True)

        # "meta.elem_idx.hj" -> "meta.data_idx.hj"
        df.rename(columns={"meta.elem_idx.hj": "meta.data_idx.hj"}, inplace=True)

    # check key
    if 0:
        nu = df.nunique()
        nu[nu == nu.max()]
    assert df["meta.data_id.hj"].is_unique, "key is not unique"
    df["key"] = df["meta.data_id.hj"]

    # sort rows
    df.sort_values("key", inplace=True, ignore_index=True)

    # sort cols
    c1 = [c for c in df.columns if not c.startswith("meta")]
    c2 = [c for c in df.columns if c.startswith("meta")]
    cols = sorted(c1) + sorted(c2)
    df = df[cols].reset_index(drop=True)
    df.sample(1).iloc[0].to_dict()

    # replace None
    df = utils.replace_blank_to_none(df)

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
    gen_format_file()  # 860.1M, 084a3e90, 1787007


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
            # python -m src.corpus.drs.format
            typer.run(main)
