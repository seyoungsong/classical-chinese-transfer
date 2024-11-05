import sys
from importlib import reload

import typer
from loguru import logger
from rich import pretty
from tqdm import tqdm

import src.crawl.wyweb_mt.root as sroot
from src import utils


def gen_format2_file() -> None:
    # read
    df = utils.read_df(sroot.PARSE2_PQ)
    df.sample(1).iloc[0].to_dict()

    # rename cols
    {k: k for k in df.columns}
    rcols = {
        "meta.row_idx": "meta.elem_idx",
    }
    df.rename(columns=rcols, inplace=True)

    # add url
    if 0:
        df["url"] = df.progress_apply(
            lambda x: f"https://github.com/NiuTrans/Classical-Modern/blob/main/双语数据/{x['meta.book_orig']}/bitext.txt#L{x['meta.elem_idx'] * 3 - 2}",
            axis=1,
        )

    # gen meta.data_id
    digit: int = df["meta.elem_idx"].apply(lambda x: len(str(x))).max()
    df["meta.elem_idx_str"] = df["meta.elem_idx"].apply(lambda x: f"L{x:0{digit}}")
    df["meta.data_id"] = df["meta.split"] + "/" + df["meta.elem_idx_str"]
    df.drop(columns=["meta.elem_idx_str", "meta.elem_idx"], inplace=True)

    # convert split (dev->valid)
    df["meta.split"].value_counts()
    df["meta.split"] = df["meta.split"].apply(lambda x: "valid" if x == "dev" else x)

    # sort cols
    c1 = [c for c in df.columns if not c.startswith("meta")]
    c2 = [c for c in df.columns if c.startswith("meta")]
    cols = sorted(c1) + sorted(c2)
    df = df[cols].reset_index(drop=True)
    df.sample(1).iloc[0].to_dict()

    # sort rows
    df.groupby(["meta.data_id"]).size().value_counts()
    df.sort_values(by=["meta.data_id"], inplace=True, ignore_index=True)

    # write
    utils.write_df2(sroot.FORMAT2_PQ, df)
    logger.debug(len(df))


def main() -> None:
    # rename and sort cols
    gen_format2_file()  # 29.5M, 16ad9a7f, 266514


if __name__ == "__main__":
    tqdm.pandas()
    #
    if hasattr(sys, "ps1"):
        pretty.install()
        reload(utils)
        reload(sroot)
    else:
        with logger.catch(onerror=lambda _: sys.exit(1)):
            # python -m src.crawl.wyweb_mt.format2
            typer.run(main)
