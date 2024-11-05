import sys
from importlib import reload

import typer
from loguru import logger
from pandarallel import pandarallel
from rich import pretty
from tqdm import tqdm

import src.dataset.mt_aug.root as sroot
from src import utils


def gen_format_file() -> None:
    # read
    df0 = utils.read_df(sroot.CONCAT_PQ)
    df = df0.copy()

    # check
    if 0:
        df1 = df.dropna()
        x1 = df1.sample(1).iloc[0].to_dict()
        utils.write_json(utils.TEMP_JSON, x1)
        utils.open_code(utils.TEMP_JSON)

    # "meta.book_id.hj" can be infered from "meta.data_id.hj"
    # exception: kzc_10308029_002, kzb
    idx = df["meta.book_id.hj"].str[1:2] == df["meta.data_id.hj"].str[1:2]
    assert idx.all(), "bad meta.book_id.hj"
    df.drop(columns=["meta.book_id.hj"], inplace=True)
    if 0:
        idx = df["meta.book_id.hj"].str[1:3] == df["meta.data_id.hj"].str[1:3]
        idx.mean()
        df[~idx].sample(1).iloc[0].to_dict()

    # "meta.elem_id.cko" -> "meta.data_id.cko"
    df.drop(columns=["meta.data_id.cko"], inplace=True)
    df.rename(columns={"meta.elem_id.cko": "meta.data_id.cko"}, inplace=True)

    # "meta.data_title.cko" -> "meta.data_date.cko"
    df.rename(columns={"meta.data_title.cko": "meta.data_date.cko"}, inplace=True)

    # "meta.elem_title.cko" -> "meta.data_title.cko"
    # "meta.elem_title.en" -> "meta.data_title.en"
    rcols = {
        "meta.elem_title.cko": "meta.data_title.cko",
        "meta.elem_title.en": "meta.data_title.en",
    }
    df.rename(columns=rcols, inplace=True)

    # "meta.elem_url.cko" -> "meta.url.cko"
    df.drop(columns=["meta.url.cko"], inplace=True)
    df.rename(columns={"meta.elem_url.cko": "meta.url.cko"}, inplace=True)

    # "meta.elem_url.en" -> "meta.data_url.en"
    df.rename(columns={"meta.elem_url.en": "meta.data_url.en"}, inplace=True)

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

    # save
    utils.write_df2(sroot.FORMAT_PQ, df)


def main() -> None:
    # rename columns, add key, sort rows, sort cols
    gen_format_file()  # 558.1M, d8d3f716, 413323


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
            # python -m src.dataset.mt_aug.format
            typer.run(main)
