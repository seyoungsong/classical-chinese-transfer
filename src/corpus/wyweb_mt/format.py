import sys
from importlib import reload

import typer
from loguru import logger
from pandarallel import pandarallel
from rich import pretty
from tqdm import tqdm

import src.corpus.wyweb_mt.root as sroot
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

    # check key
    if 0:
        nu = df.nunique()
        nu[nu == nu.max()]
    assert df["meta.data_id.cc"].is_unique, "key is not unique"
    df["key"] = df["meta.data_id.cc"]

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
    gen_format_file()


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
            # python -m src.corpus.wyweb_mt.format
            typer.run(main)
