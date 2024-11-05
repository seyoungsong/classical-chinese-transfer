import sys
from importlib import reload

import typer
from loguru import logger
from rich import pretty
from tqdm import tqdm

import src.crawl.klc_hj_ko.root as sroot
import src.tool.eval as etool
from src import utils


def gen_format2_file() -> None:
    # init
    tqdm.pandas()

    # read
    df = utils.read_df(sroot.FORMAT_PQ)
    df.sample(1).iloc[0].to_dict()

    # text_xml to text
    if 0:
        idx = df["text_xml"].apply(lambda x: "</" in str(x))
        x1 = df[idx].sample(1).iloc[0].to_dict()
        s_xml = x1["text_xml"]
        etool.xml2items(s_xml=s_xml)
        etool.xml2plaintext(s_xml=s_xml)
        utils.open_url(x1["meta.url"])
    df["text_xml"].fillna("", inplace=True)
    df["text"] = df["text_xml"].progress_apply(etool.xml2plaintext)
    df = utils.replace_blank_to_none(df)

    # sort columns
    c1 = [c for c in df.columns if not c.startswith("meta")]
    c2 = [c for c in df.columns if c.startswith("meta")]
    cols = sorted(c1) + sorted(c2)
    df = df[cols].reset_index(drop=True)

    # write
    df.sample(1).iloc[0].to_dict()
    utils.write_df2(sroot.FORMAT2_PQ, df)


def main() -> None:
    # change format
    gen_format2_file()  # 644.2M, 3fd276e7


if __name__ == "__main__":
    if hasattr(sys, "ps1"):
        pretty.install()
        reload(utils)
        reload(sroot)
    else:
        with logger.catch(onerror=lambda _: sys.exit(1)):
            # python -m src.crawl.klc_hj_ko.format2
            typer.run(main)
