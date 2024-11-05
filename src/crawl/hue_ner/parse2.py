import sys
from ast import literal_eval
from importlib import reload

import typer
from loguru import logger
from pandarallel import pandarallel
from rich import pretty
from tqdm import tqdm

import src.crawl.hue_ner.root as sroot
import src.tool.corpus as ctool
import src.tool.eval as etool
from src import utils


def gen_text_xml(text: str, label: str) -> str:
    ner_tags: list[str] = literal_eval(label)
    assert len(text) == len(ner_tags), "len mismatch: text, ner_tags"
    s_xml = etool.iob2xml(tokens=list(text), ner_tags=ner_tags)
    return s_xml


def gen_parse2_file() -> None:
    # read
    df = utils.read_df(sroot.CLEAN_PQ)
    df.sample(1).iloc[0].to_dict()

    # test
    if 0:
        x1 = df.sample(1).iloc[0].to_dict()
        x1
        text = x1["text"]
        label = x1["label"]
        s_xml = gen_text_xml(text=text, label=label)
        s_xml
        etool.xml2plaintext(s_xml=s_xml)

    # parse
    df["text_xml"] = df.parallel_apply(  # type: ignore
        lambda x1: gen_text_xml(text=x1["text"], label=x1["label"]), axis=1
    )
    df.drop(columns=["text", "label"], inplace=True)
    df["text"] = df["text_xml"].progress_apply(lambda x: etool.xml2plaintext(s_xml=x))
    df.sample(1).iloc[0].to_dict()

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

    # save
    utils.write_df2(sroot.PARSE2_PQ, df)


def main() -> None:
    # parse body html to text

    # init
    tqdm.pandas()
    pandarallel.initialize(progress_bar=True)
    #
    gen_parse2_file()  # 113.6M, 6c743793


if __name__ == "__main__":
    if hasattr(sys, "ps1"):
        pretty.install()
        reload(utils)
        reload(sroot)
        reload(ctool)
        #
        tqdm.pandas()
        pandarallel.initialize(progress_bar=True)
    else:
        with logger.catch(onerror=lambda _: sys.exit(1)):
            # python -m src.crawl.hue_ner.parse2
            typer.run(main)
