import sys
from importlib import reload

import typer
from loguru import logger
from pandarallel import pandarallel
from rich import pretty
from tqdm import tqdm

import src.corpus.wyweb_punc.root as sroot
import src.tool.data as dtool
from src import utils


def gen_normal_file() -> None:
    # read
    df0 = utils.read_df(sroot.CLEAN_PQ)
    df = df0.copy()

    # sample
    if 0:
        df1 = df.dropna()
        x1 = df1.sample(1).iloc[0].to_dict()
        utils.write_json(utils.TEMP_JSON, x1)
        utils.open_code(utils.TEMP_JSON)

    # normalize
    text_cols = [c for c in df.columns if c.startswith("text") and "xml" not in c]
    xml_cols = [c for c in df.columns if c.startswith("text") and "xml" in c]
    for c in tqdm(xml_cols):
        df[c] = df[c].parallel_apply(dtool.normalize_xml)
    for c in tqdm(text_cols):
        df[c] = df[c].parallel_apply(dtool.normalize_str)

    # save
    utils.write_df2(sroot.NORMAL_PQ, df)


def main() -> None:
    # normalization (cjk, punc)
    gen_normal_file()


if __name__ == "__main__":
    tqdm.pandas()
    pandarallel.initialize(progress_bar=True)
    #
    if hasattr(sys, "ps1"):
        pretty.install()
        reload(utils)
        reload(sroot)
        reload(dtool)
    else:
        with logger.catch(onerror=lambda _: sys.exit(1)):
            # python -m src.corpus.wyweb_punc.normal
            typer.run(main)
