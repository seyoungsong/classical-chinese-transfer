import sys
from importlib import reload

import typer
from loguru import logger
from pandarallel import pandarallel
from rich import pretty
from tqdm import tqdm

import src.corpus.klc_hj.root as sroot
import src.tool.data as dtool
from src import utils


def merge_split(split1: str, split2: str) -> str:
    # split2 has priority
    vals = {"train", "valid", "test"}
    assert split1 in vals and split2 in vals
    if split2 == "train":
        return split1
    elif split2 == "test":
        return f"{split1}2"
    else:
        raise ValueError(f"{split1=} {split2=}")


def gen_filter2_file() -> None:
    # read
    df0 = utils.read_df(sroot.FILTER_PQ)
    df = df0.copy()
    df.sample(1).iloc[0].to_dict()

    # gen book_id (ITKC_MP_0001A_0010_000_0010 -> ITKC_MP_0001A)
    df["book_id"] = df["meta.data_id.hj"].progress_apply(
        lambda x: "_".join(str(x).split("_")[:3])
    )
    if 0:
        df = df.sample(1000)
        df.groupby(["meta.book_title.hj", "book_id"]).size().sort_values()
        df.groupby(["meta.book_title.hj", "book_id"]).value_counts().value_counts()

    # split2 from book_id
    df["split2"] = df["book_id"].parallel_apply(
        lambda x: dtool.uid2split(uid=x, ratio=(0.5, 0.0, 0.5))
    )
    if 0:
        df["split2"].value_counts() / len(df) * 100
        df["split2"].value_counts()

    # merge split2 to split
    df["split3"] = df.progress_apply(  # type: ignore
        lambda x: merge_split(split1=x["split"], split2=x["split2"]), axis=1
    )
    if 0:
        df["split"].value_counts() / len(df) * 100
        df["split2"].value_counts() / len(df) * 100
        df["split3"].value_counts() / len(df) * 100

    # drop and rename cols
    df["split"] = df["split3"]
    df.drop(columns=["split2", "split3", "book_id"], inplace=True)

    # sort cols
    c1 = [c for c in df.columns if not c.startswith("meta")]
    c2 = [c for c in df.columns if c.startswith("meta")]
    cols = sorted(c1) + sorted(c2)
    df = df[cols].reset_index(drop=True)
    df.sample(1).iloc[0].to_dict()

    # check
    logger.debug(round(df["split"].value_counts().sort_index() / len(df) * 100, 1))
    """
test       5.2
test2      4.8
train     41.6
train2    38.4
valid      5.2
valid2     4.8
    """

    # save
    utils.write_df2(sroot.FILTER2_PQ, df)


def main() -> None:
    # overwrite split to match experimental setting
    gen_filter2_file()  # 393.9M, ef336ee8, 652622


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
            # python -m src.corpus.klc_hj.filter2
            typer.run(main)
