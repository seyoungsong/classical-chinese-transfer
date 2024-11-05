import sys
from importlib import reload

import typer
from loguru import logger
from rich import pretty
from tqdm import tqdm

import src.crawl.drs_hj.root as sroot
import src.tool.eval as etool
from src import utils


def gen_format2_file() -> None:  # noqa: C901
    # init
    tqdm.pandas()

    # read
    df = utils.read_df(sroot.FORMAT_PQ)
    df.sample(1).iloc[0].to_dict()

    # text_xml to text
    if 0:
        idx = df["text_xml"].apply(lambda x: "</" in str(x))
        idx.sum()
        idx.mean()
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
        temp1 = df1["text_xml"].sample(n=10000).progress_apply(etool.names_in_xml)
        logger.debug(sorted(set(sum(temp1, []))))
        s1 = "\n\n".join(df1["text_xml"].to_list())
        utils.write_str(utils.TEMP_TXT, s1)

    df.fillna("", inplace=True)
    df["text"] = df["text_xml"].progress_apply(etool.xml2plaintext)
    df = utils.replace_blank_to_none(df)
    if 0:
        for s_xml in tqdm(df["text_xml"]):
            _ = etool.xml2plaintext(s_xml=s_xml)
        x1 = df[df["text_xml"] == s_xml].iloc[0].to_dict()

    if 0:
        # drop text_xml if no valid NER tags
        df.drop(columns=["text_xml", "meta.book_extra_orig"], inplace=True)

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

    # check data_id_ko
    df["meta.data_id_ko"].fillna("", inplace=True)
    temp1 = df["meta.data_id_ko"].apply(len)
    vc = temp1.value_counts()
    sum(vc[vc < 1000].values)  # 25
    bad_lens = vc[vc < 100].index
    {
        i: df[temp1 == i][["meta.data_id_ko", "meta.url"]].to_dict(orient="records")
        for i in bad_lens
    }
    # _ 대신 -를 사용한 경우, 끝에 _가 붙은 경우 등

    # drop empty rows
    df["text"].isna().sum()  # 16
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

    # drop duplicate columns
    df.sample(1).iloc[0].to_dict()
    if 0:
        if df["meta.elem_url"].equals(df["meta.url"]):
            df.drop(columns=["meta.elem_url"], inplace=True)
        if df["meta.data_title_mokcha"].equals(df["meta.data_title"]):
            df.drop(columns=["meta.data_title_mokcha"], inplace=True)

    # check
    if 0:
        df["meta.book_extra"].value_counts() / len(df) * 100

    # write
    df.sample(1).iloc[0].to_dict()
    utils.write_df2(sroot.FORMAT2_PQ, df)
    logger.debug(f"len(df)={len(df)}")


def main() -> None:
    # convert text_xml to text and drop empty columns
    gen_format2_file()  # 697.9M, ee58798f


if __name__ == "__main__":
    if hasattr(sys, "ps1"):
        pretty.install()
        reload(utils)
        reload(sroot)
        reload(etool)
    else:
        with logger.catch(onerror=lambda _: sys.exit(1)):
            # python -m src.crawl.drs_hj.format2
            typer.run(main)
