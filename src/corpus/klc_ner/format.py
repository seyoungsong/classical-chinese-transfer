import sys
from collections import Counter
from importlib import reload

import pandas as pd
import typer
from loguru import logger
from pandarallel import pandarallel
from rich import pretty
from tqdm import tqdm

import src.corpus.klc_ner.root as sroot
import src.tool.eval as etool
from src import utils


def xml_change_name(s_xml: str, rule: dict[str, str]) -> str:
    items = etool.xml2items(s_xml)
    items2 = [dict(d, name=rule.get(d["name"], None)) for d in items]  # type: ignore
    s_xml2 = etool.items2xml(items2)
    return s_xml2


def change_ner_tags(df: pd.DataFrame) -> pd.DataFrame:
    if 0:
        x1 = df.sample(1).iloc[0].to_dict()
        s_xml = x1["text_xml.hj"]
        s_xml

    # count
    counter = Counter()  # type: ignore
    for s_xml in tqdm(df["text_xml.hj"]):
        items = etool.xml2items(s_xml)
        names = [d["name"] for d in items if d["name"]]
        counter.update(names)

    # check
    df1 = pd.DataFrame(counter.most_common(), columns=["name", "count"])
    df1["percent"] = df1["count"] / df1["count"].sum() * 100
    df1["percent"] = df1["percent"].round(2)
    df1
    """
                                                name   count  percent
0                                             5fb636  402601    92.79
1                            948a54;font-size:0.9em;   14481     3.34  # 금색. 뭔지 모르겠음.
2   padding:0 3px;background-color:black;color:#fff;    7556     1.74
3  padding:0 3px;background-color:#ceb443;color:#...    5127     1.18
4                 background-color:black;color:white    3289     0.76
5    padding:0 3px;background-color:grey;color:#fff;     832     0.19
    """

    # check
    if 0:
        name1 = "948a54"
        idx = df["text_xml.hj"].str.contains(name1, regex=False)
        x1 = df[idx].sample(1).iloc[0].to_dict()
        s_xml = x1["text_xml.hj"]
        xml_change_name(s_xml, {"948a54;font-size:0.9em;": "other"})
        xml_change_name(s_xml, {"5fb636": "other"})

    # we merge 5fb636 to other, and anything else to None (remove)
    rule = {"5fb636": "other"}
    df["text_xml.hj"] = df["text_xml.hj"].progress_apply(
        lambda x: xml_change_name(s_xml=x, rule=rule)
    )
    df["text_xml.ko"] = df["text_xml.ko"].progress_apply(
        lambda x: xml_change_name(s_xml=x, rule=rule)
    )

    # count
    counter = Counter()
    for s_xml in tqdm(df["text_xml.hj"]):
        items = etool.xml2items(s_xml)
        names = [d["name"] for d in items if d["name"]]
        counter.update(names)
    # check
    df1 = pd.DataFrame(counter.most_common(), columns=["name", "count"])
    df1["percent"] = df1["count"] / df1["count"].sum() * 100
    df1["percent"] = df1["percent"].round(2)
    df1
    """
    name   count  percent
0  other  402601    100.0
    """

    return df


def gen_format_file() -> None:
    # read
    df0 = utils.read_df(sroot.ALIGN_PQ)
    df = df0.copy()

    # drop those without ner tags
    idx = df["text_xml.hj"].str.contains(utils.NER_PREF, regex=False)
    round(idx.mean() * 100, 2)
    df = df[idx].reset_index(drop=True)

    # check and fix xml tags
    df = change_ner_tags(df)

    # drop those without ner tags
    idx = df["text_xml.hj"].str.contains(utils.NER_PREF, regex=False)
    round(idx.mean() * 100, 2)
    df = df[idx].reset_index(drop=True)

    # sort rows
    df.sort_values("key", inplace=True, ignore_index=True)

    # replace None
    df = utils.replace_blank_to_none(df)

    # sort cols
    c1 = [c for c in df.columns if not c.startswith("meta")]
    c2 = [c for c in df.columns if c.startswith("meta")]
    cols = sorted(c1) + sorted(c2)
    df = df[cols].reset_index(drop=True)
    df.sample(1).iloc[0].to_dict()

    # check
    if 0:
        df1 = df.dropna()
        x1 = df1.sample(1).iloc[0].to_dict()
        utils.write_json(utils.TEMP_JSON, x1)
        utils.open_code(utils.TEMP_JSON)

    # save
    utils.write_df2(sroot.FORMAT_PQ, df)


def main() -> None:
    # rename columns, add key, sort rows, sort cols
    gen_format_file()


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
            # python -m src.corpus.klc_ner.format
            typer.run(main)
