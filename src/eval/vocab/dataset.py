import sys
from importlib import reload

import typer
from loguru import logger
from pandarallel import pandarallel
from rich import pretty
from tqdm import tqdm

import src.dataset.vocab.root
import src.eval.vocab.root as sroot
import src.tool.eval as etool
from src import utils


def gen_dataset_file() -> None:
    # read file
    df = utils.read_df(src.dataset.vocab.root.CONCAT_PQ)
    df.sample(1).iloc[0].to_dict()

    # check: stat
    df.info()
    sz1 = (
        df.groupby(["lang", "meta.corpus"], dropna=False)
        .size()
        .reset_index(name="count")  # type: ignore
    )
    sz2 = (
        df.groupby(["lang", "meta.corpus"], dropna=False)
        .agg({"text": lambda x: x.str.len().sum()})
        .reset_index()
    )
    sz3 = sz1.merge(sz2, on=["lang", "meta.corpus"], how="inner")
    sz3["avg"] = sz3["text"] / sz3["count"]
    sroot.RESULT_DIR.mkdir(exist_ok=True, parents=True)
    utils.write_df(sroot.RESULT_DIR / "size_orig.tsv", sz3)

    # remove punc
    if 0:
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        df["text"] = df["text"].parallel_apply(etool.remove_punc)
        df.sort_values(by="key2", inplace=True, ignore_index=True)

    # drop empty text
    df = utils.replace_blank_to_none(df)
    df.dropna(subset=["text"], inplace=True)
    df.dropna(axis=1, how="all", inplace=True)

    # check: no empty text
    if 0:
        cols = [c for c in df.columns if not c.startswith("meta")]
        assert df[cols].isnull().sum().sum() == 0, "no empty text"

    # save
    sroot.DATASET_PQ.parent.mkdir(exist_ok=True, parents=True)
    utils.write_df2(sroot.DATASET_PQ, df)


def main() -> None:
    gen_dataset_file()  # 3.6G, 04a0e230, 5963484


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
            # python -m src.eval.vocab.dataset
            typer.run(main)
