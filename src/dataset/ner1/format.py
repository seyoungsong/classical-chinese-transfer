import sys
from collections import Counter
from importlib import reload

import pandas as pd
import typer
from loguru import logger
from pandarallel import pandarallel
from rich import pretty
from tqdm import tqdm

import src.dataset.ner1.root as sroot
import src.dataset.ner.root
import src.tool.eval as etool
from src import utils


def xml_change_name(s_xml: str) -> str:
    items = etool.xml2items(s_xml)
    items2 = [dict(d, name="other") if d["name"] is not None else d for d in items]
    s_xml2 = etool.items2xml(items2)
    return s_xml2


def change_ner_tags(df: pd.DataFrame) -> pd.DataFrame:
    if 0:
        x1 = df.sample(1).iloc[0].to_dict()
        s_xml = x1["text_xml"]
        s_xml
        xml_change_name(s_xml=s_xml)

    # we merge all NER labels to 'other'
    df["text_xml"] = df["text_xml"].progress_apply(xml_change_name)

    # count
    counter = Counter()  # type: ignore
    for s_xml in tqdm(df["text_xml"]):
        items = etool.xml2items(s_xml)
        names = [d["name"] for d in items if d["name"]]
        counter.update(names)
    # check
    df1 = pd.DataFrame(counter.most_common(), columns=["name", "count"])
    df1["percent"] = df1["count"] / df1["count"].sum() * 100
    df1["percent"] = df1["percent"].round(2)
    df1
    """
    name    count  percent
0  other  2680643    100.0
    """

    return df


def gen_format_file() -> None:
    # read
    df0 = utils.read_df(src.dataset.ner.root.FILTER_PQ)
    df = df0.copy()
    df.sample(1).iloc[0].to_dict()

    # change NER tags
    df = change_ner_tags(df)

    # save
    utils.write_df2(sroot.FORMAT_PQ, df)


def main() -> None:
    # change NER tags to 'other'
    gen_format_file()  # 337.3M, 14f3b397, 410382


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
            # python -m src.dataset.ner1.format
            typer.run(main)
