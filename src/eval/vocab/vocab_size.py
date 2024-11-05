import sys
from importlib import reload

import typer
from loguru import logger
from pandarallel import pandarallel
from rich import pretty
from tqdm import tqdm

import src.eval.vocab.root as sroot
from src import utils


def gen_vocab_size_tsv() -> None:
    # read
    df = utils.read_df(sroot.CHAR_COUNT2_PQ)
    df.sample(1).iloc[0].to_dict()

    # unique char per corpus and lang
    df1 = df.drop_duplicates(subset=["lang", "char"], ignore_index=True)
    df2 = df1.groupby(["lang"]).size().reset_index(name="count")  # type: ignore
    fname = sroot.RESULT_DIR / "vocab_size.tsv"
    fname.parent.mkdir(exist_ok=True, parents=True)
    utils.write_df(fname, df2)


def main() -> None:
    gen_vocab_size_tsv()


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
            # python -m src.eval.vocab.vocab_size
            typer.run(main)
