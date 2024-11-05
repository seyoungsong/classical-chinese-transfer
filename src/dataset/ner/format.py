import sys
from collections import Counter
from importlib import reload

import pandas as pd
import typer
from loguru import logger
from pandarallel import pandarallel
from rich import pretty
from tqdm import tqdm

import src.dataset.ner.root as sroot
import src.tool.eval as etool
from src import utils


def xml_change_name(s_xml: str, corpus_name: dict[str, str]) -> str:
    items = etool.xml2items(s_xml)
    items2 = [dict(d, name=corpus_name + "_" + d["name"]) if d["name"] is not None else d for d in items]  # type: ignore
    s_xml2 = etool.items2xml(items2)
    return s_xml2


def change_ner_tags(df: pd.DataFrame) -> pd.DataFrame:
    if 0:
        x1 = df.sample(1).iloc[0].to_dict()
        s_xml = x1["text_xml"]
        s_xml

    # count
    counter = Counter()  # type: ignore
    for s_xml, corpus in tqdm(zip(df["text_xml"], df["meta.corpus"]), total=len(df)):
        items = etool.xml2items(s_xml)
        names = [f"{corpus}_" + d["name"] for d in items if d["name"]]
        counter.update(names)

    # check
    df1 = pd.DataFrame(counter.most_common(), columns=["name", "count"])
    df1["percent"] = df1["count"] / df1["count"].sum() * 100
    df1["percent"] = df1["percent"].round(2)
    df1

    """
                 name    count  percent
0          ajd_person  1507242    56.05
1        ajd_location   499296    18.57
2           klc_other   379976    14.13
3     wyweb_ner_other   179451     6.67
4  wyweb_ner_bookname    68975     2.56
5           ajd_other    54240     2.02
    """

    # we merge corpus names in front of NER tags
    df["meta.corpus"].value_counts()
    df["corpus"] = df["meta.corpus"].replace({"wyweb_ner": "wyweb"})
    df["corpus"].value_counts()
    if 0:
        x = df.sample(1).iloc[0].to_dict()
        s_xml = x["text_xml"]
        corpus_name = x["corpus"]
        xml_change_name(s_xml=s_xml, corpus_name=corpus_name)
    df["text_xml"] = df.progress_apply(  # type: ignore
        lambda x: xml_change_name(s_xml=x["text_xml"], corpus_name=x["corpus"]),
        axis=1,
    )
    df.drop(columns=["corpus"], inplace=True)

    # count
    counter = Counter()
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
0      ajd_person  1507242    56.05
1    ajd_location   499296    18.57
2       klc_other   379976    14.13
3     wyweb_other   179451     6.67
4  wyweb_bookname    68975     2.56
5       ajd_other    54240     2.02
    """

    return df


def gen_format_file() -> None:
    # read
    df0 = utils.read_df(sroot.CONCAT_PQ)
    df = df0.copy()
    df.sample(1).iloc[0].to_dict()

    # change NER tags
    df = change_ner_tags(df)

    # check
    temp1: utils.SeriesType = df["text_xml"].progress_apply(etool.is_valid_xml)
    temp1.value_counts()
    assert temp1.all(), "some xml are invalid"

    # save
    utils.write_df2(sroot.FORMAT_PQ, df)


def main() -> None:
    # change NER tags to include corpus prefix
    gen_format_file()  # 340.7M, 5333df8b, 453584


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
            # python -m src.dataset.ner.format
            typer.run(main)
