import sys
from importlib import reload

import typer
from loguru import logger
from pandarallel import pandarallel
from rich import pretty
from tqdm import tqdm

import src.crawl.wyweb_punc.root as sroot
import src.tool.corpus as ctool
import src.tool.eval as etool
from src import utils


def gen_text_xml(text: str, label: str) -> str:
    assert len(text) == len(label), "len mismatch"
    pairs = list(zip(text, label, strict=True))
    items = []
    for pair1 in pairs:
        c1, p1 = pair1
        item1 = {"text": c1, "name": None}
        items.append(item1)
        if p1 != "O":
            item2 = {"text": p1, "name": "punc"}
            items.append(item2)  # type: ignore
    s_xml = etool.items2xml(items=items)
    if 0:
        etool.xml2plaintext(s_xml=s_xml)
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
    assert (df["text"].str.len() == df["label"].str.len()).all(), "len mismatch"
    df["text_xml"] = df.progress_apply(  # type: ignore
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
    gen_parse2_file()  # 60.2M, cf35d00a


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
            # python -m src.crawl.wyweb_punc.parse2
            typer.run(main)
