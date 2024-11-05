import sys
from collections import Counter
from importlib import reload

import pandas as pd
import typer
from loguru import logger
from pandarallel import pandarallel
from rich import pretty
from tqdm import tqdm

import src.corpus.ajd_ner.root as sroot
import src.tool.eval as etool
from src import utils


def xml_change_name(s_xml: str, rule: dict[str, str]) -> str:
    items = etool.xml2items(s_xml)
    items2 = [dict(d, name=rule.get(d["name"], d["name"])) for d in items]  # type: ignore
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
     name    count  percent
0  person  1507242    73.14
1   place   499296    24.23
2    book    49180     2.39
3     era     5060     0.25
    """

    # we merge book and era to other, and rename place to location
    rule = {"book": "other", "era": "other", "place": "location"}
    df["text_xml.hj"] = df["text_xml.hj"].progress_apply(
        lambda x: xml_change_name(s_xml=x, rule=rule)
    )
    df["text_xml.oko"] = df["text_xml.oko"].progress_apply(
        lambda x: xml_change_name(s_xml=x, rule=rule)
    )
    """
       name    count  percent
0    person  1507242    73.14
1  location   499296    24.23
2     other    54240     2.63
    """

    return df


def gen_format_file() -> None:
    # read
    df0 = utils.read_df(sroot.ALIGN_PQ)
    df = df0.copy()

    # check
    if 0:
        df1 = df.dropna()
        x1 = df1.sample(1).iloc[0].to_dict()
        utils.write_json(utils.TEMP_JSON, x1)
        utils.open_code(utils.TEMP_JSON)

    # check and fix xml tags
    df = change_ner_tags(df)

    # check key
    if 0:
        nu = df.nunique()
        nu[nu == nu.max()]
    if "key" not in df.columns:
        assert df["meta.data_id.hj"].is_unique, "key is not unique"
        df["key"] = df["meta.data_id.hj"]
    else:
        logger.debug("key exists")
        assert df["key"].is_unique, "key is not unique"

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
            # python -m src.corpus.ajd_ner.format
            typer.run(main)
