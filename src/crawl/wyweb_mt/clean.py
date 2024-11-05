import sys
from importlib import reload
from pathlib import Path

import typer
from loguru import logger
from rich import pretty

import src.crawl.wyweb_mt.root as sroot
from src import utils


def gen_path(fname: str) -> str:
    p1 = Path(fname).resolve()
    p2 = p1.relative_to(sroot.CRAWL_DL_DIR)
    path1 = p2.parent
    return str(path1)


def gen_clean_file() -> None:
    # read
    df = utils.read_df(sroot.PARSE_PQ)
    df.sample(1).iloc[0].to_dict()
    df.info()

    # convert
    df["split"] = df["x1.fname"].apply(lambda x: Path(x).stem)
    df.drop(columns=["x1.fname"], inplace=True)

    # drop
    dcols = ["x1.temp_id", "x1.fname2", "x1.size", "x1.temp_len"]
    df.drop(columns=[c for c in dcols if c in df.columns], inplace=True)

    # rename
    {k: f"meta.{k}" for k in df.columns}
    rcols = {
        "cc": "text_cc",
        "zh": "text_zh",
        "row_idx": "meta.row_idx",
        "x.fname": "meta.fname",
        "split": "meta.split",
    }
    assert len(rcols.values()) == len(set(rcols.values()))
    df.rename(columns=rcols, inplace=True)

    # drop cols
    dcols = [c for c in df.columns if c.startswith("_")]
    df.drop(columns=dcols, inplace=True)

    # sort columns
    c1 = [c for c in df.columns if not c.startswith("meta")]
    c2 = [c for c in df.columns if c.startswith("meta")]
    cols = sorted(c1) + sorted(c2)
    df = df[cols].reset_index(drop=True)
    df.sample(1).iloc[0].to_dict()

    # sort rows
    cols = ["meta.split", "meta.row_idx"]
    df.groupby(cols).size().value_counts()
    df.sort_values(cols, inplace=True, ignore_index=True)

    # fix types
    df["meta.row_idx"] = df["meta.row_idx"].astype(int)
    df.sample().iloc[0].to_dict()

    # empty to nan
    df.replace(r"^\s*$", None, regex=True, inplace=True)
    df.info()
    df.isna().sum()[df.isna().sum() > 0]
    df.fillna("", inplace=True)

    # add: path
    if 0:
        x1 = df.sample(1).iloc[0].to_dict()
        gen_path(fname=x1["meta.fname"])
        df["meta.path"] = df["meta.fname"].parallel_apply(gen_path)
        df.drop(columns=["meta.fname"], inplace=True)

    # save
    utils.write_df2(sroot.CLEAN_PQ, df)


def main() -> None:
    # remove errors and drop unnecessary columns
    gen_clean_file()  # 29.8M, 53d522f1


if __name__ == "__main__":
    if hasattr(sys, "ps1"):
        pretty.install()
        reload(utils)
        reload(sroot)
    else:
        with logger.catch(onerror=lambda _: sys.exit(1)):
            # python -m src.crawl.wyweb_mt.clean
            typer.run(main)
