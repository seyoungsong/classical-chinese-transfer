import json
import sys
from importlib import reload

import typer
from loguru import logger
from rich import pretty
from tqdm import tqdm

import src.crawl.ocdb_cc_ko.root as sroot
from src import utils


def gen_format_file() -> None:
    # read
    df = utils.read_df(sroot.PARSE2_PQ)
    df.sample(1).iloc[0].to_dict()

    # rename cols
    {k: k for k in df.columns}
    rcols = {
        "title.text": "meta.title",
        "meta.body.text": "text",
        "meta.bookname": "meta.bookname",
        "meta.data_id": "meta.data_id",
        "meta.data_url": "meta.data_url",
        "meta.url": "meta.url",
    }
    df.rename(columns=rcols, inplace=True)

    # unpack text
    df["text.cc"] = df.progress_apply(  # type: ignore
        lambda x: json.dumps(json.loads(x["text"])["cc"], ensure_ascii=False), axis=1
    )
    df["text.ko"] = df.progress_apply(  # type: ignore
        lambda x: json.dumps(json.loads(x["text"])["ko"], ensure_ascii=False), axis=1
    )
    df.drop(columns=["text"], inplace=True)

    # sort
    c1 = [c for c in df.columns if not c.startswith("meta")]
    c2 = [c for c in df.columns if c.startswith("meta")]
    cols = sorted(c1) + sorted(c2)
    df = df[cols].reset_index(drop=True)
    df.sample(1).iloc[0].to_dict()

    # write
    utils.write_df2(sroot.FORMAT_PQ, df)


def main() -> None:
    # change format
    gen_format_file()  # 29.6M, c8253283, 28341


if __name__ == "__main__":
    tqdm.pandas()
    #
    if hasattr(sys, "ps1"):
        pretty.install()
        reload(utils)
        reload(sroot)
    else:
        with logger.catch(onerror=lambda _: sys.exit(1)):
            # python -m src.crawl.ocdb_cc_ko.format
            typer.run(main)
