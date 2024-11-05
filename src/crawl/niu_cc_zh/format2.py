import sys
from importlib import reload

import typer
from loguru import logger
from rich import pretty
from tqdm import tqdm

import src.crawl.niu_cc_zh.root as sroot
import src.tool.eval as etool
from src import utils


def gen_format2_file() -> None:  # noqa: C901
    # init
    tqdm.pandas()

    # read
    df = utils.read_df(sroot.FORMAT_PQ)
    df.sample(1).iloc[0].to_dict()

    # add url
    # https://github.com/NiuTrans/Classical-Modern/blob/main/双语数据/三国志/蜀书/后主传/bitext.txt#L8
    df["url"] = df.progress_apply(  # type: ignore
        lambda x: f"https://github.com/NiuTrans/Classical-Modern/blob/main/双语数据/{x['meta.book_orig']}/bitext.txt#L{x['meta.elem_idx'] * 3 - 2}",
        axis=1,
    )

    # gen meta.data_id
    digit: int = df["meta.elem_idx"].apply(lambda x: len(str(x))).max()
    df["meta.elem_idx_str"] = df["meta.elem_idx"].apply(lambda x: f"L{x:0{digit}}")
    df["meta.data_id"] = df["meta.book_orig"] + "/" + df["meta.elem_idx_str"]
    df.drop(
        columns=["meta.elem_idx_str", "meta.elem_idx", "meta.book_orig"], inplace=True
    )

    # sort columns
    c1 = [c for c in df.columns if not c.startswith("meta")]
    c2 = [c for c in df.columns if c.startswith("meta")]
    cols = sorted(c1) + sorted(c2)
    df = df[cols].reset_index(drop=True)

    # sort rows
    df.sort_values(by=["meta.data_id"], inplace=True, ignore_index=True)

    # write
    df.sample(1).iloc[0].to_dict()
    utils.write_df2(sroot.FORMAT2_PQ, df)
    logger.debug(f"len(df)={len(df)}")


def main() -> None:
    # convert text_xml to text and drop empty columns
    gen_format2_file()  # 92.5M, 8fb1f5bd, 972467


if __name__ == "__main__":
    if hasattr(sys, "ps1"):
        pretty.install()
        reload(utils)
        reload(sroot)
        reload(etool)
    else:
        with logger.catch(onerror=lambda _: sys.exit(1)):
            # python -m src.crawl.niu_cc_zh.format2
            typer.run(main)
