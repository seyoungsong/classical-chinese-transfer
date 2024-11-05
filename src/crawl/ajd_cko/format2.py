import re
import sys
from importlib import reload

import typer
from loguru import logger
from rich import pretty
from tqdm import tqdm

import src.crawl.ajd_cko.root as sroot
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
        idx = df["text_xml"].apply(lambda x: "font-size" in str(x))
        idx = df["text_xml"].apply(lambda x: utils.NER_PREF in str(x))
        idx.sum()
        df1 = df[idx].reset_index(drop=True)
        len(df1)
        #
        x1 = df1.sample(1).iloc[0].to_dict()
        s_xml = x1["text_xml"]
        etool.xml2items(s_xml=s_xml)
        etool.xml2plaintext(s_xml=s_xml)
        utils.open_url(x1["meta.url"])
        etool.names_in_xml(s_xml=s_xml)
        #
        temp1 = df1["text_xml"].progress_apply(etool.names_in_xml)
        logger.debug(sorted(set(sum(temp1, []))))
        s1 = "\n\n".join(df1["text_xml"].to_list())
        utils.write_str(utils.TEMP_TXT, s1)

    df["text_xml"].fillna("", inplace=True)
    df["text"] = df["text_xml"].progress_apply(etool.xml2plaintext)
    df = utils.replace_blank_to_none(df)

    # drop text_xml since no valid NER tags
    df.drop(columns=["text_xml"], inplace=True)

    # fix: corpus-specific artifacts
    # e.g. AJD: 【원전】 3 집 494 면\n【분류】 인사-임면(任免)
    pat = re.compile(r"\s*【원전】[\s\S]*【분류】[\s\S]*")
    if 0:
        s1 = df["text"].sample(1).iloc[0]
        pat.findall(s1)
        pat.sub("", s1)
    df["text"] = df["text"].progress_apply(lambda x: pat.sub("", x))

    # sort columns
    c1 = [c for c in df.columns if not c.startswith("meta")]
    c2 = [c for c in df.columns if c.startswith("meta")]
    cols = sorted(c1) + sorted(c2)
    df = df[cols].reset_index(drop=True)

    # fillna
    df.info()
    df.isna().sum()[df.isna().sum() > 0]
    df.sample(1).iloc[0].to_dict()
    if 0:
        for col in ["meta.data_title_mokcha", "meta.elem_dci"]:
            df[col].fillna("N/A", inplace=True)

    # drop empty rows
    df["text"].isna().sum()  # 0
    if 0:
        df1 = df[df["text"].isna()].reset_index(drop=True)
        df1.sample(1).iloc[0].to_dict()  # 좌목, 낙장(원문 훼손) 등
        df1["meta.elem_title"].value_counts()
    if 0:
        # 일성록은 body가 없는 경우가 있어서 skip
        df = df.dropna(subset=["text"]).reset_index(drop=True)

    # drop trivial columns
    df.nunique()
    if 0:
        df["meta.data_title_mokcha"].value_counts()
    dcols = ["lang", "punc_type", "meta.data_title_mokcha"]
    dcols = [c for c in dcols if c in df.columns]
    df.drop(columns=dcols, inplace=True)

    # check
    if 0:
        df["meta.book_extra"].value_counts() / len(df) * 100

    # write
    df.sample(1).iloc[0].to_dict()
    utils.write_df2(sroot.FORMAT2_PQ, df)
    logger.debug(f"len(df)={len(df)}")


def main() -> None:
    # convert text_xml to text and drop empty columns
    gen_format2_file()  # 22.2M, 5dff651c, 51376


if __name__ == "__main__":
    if hasattr(sys, "ps1"):
        pretty.install()
        reload(utils)
        reload(sroot)
        reload(etool)
    else:
        with logger.catch(onerror=lambda _: sys.exit(1)):
            # python -m src.crawl.ajd_cko.format2
            typer.run(main)
