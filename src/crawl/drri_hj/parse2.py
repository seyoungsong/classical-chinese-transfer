import random
import sys
import uuid
from importlib import reload
from typing import Any

import pandas as pd
import typer
from bs4 import BeautifulSoup, Tag
from loguru import logger
from pandarallel import pandarallel
from rich import pretty
from tqdm import tqdm

import src.crawl.drri_hj.root as sroot
import src.tool.corpus as ctool
import src.tool.eval as etool
from src import utils

_PREF = utils.NER_PREF


def parse_elem_both_html(  # noqa: C901
    x: dict[str, Any], col: str = "meta.elem_body_html"
) -> str:
    html: str = x[col]
    soup = BeautifulSoup(html, "xml")
    if 0:
        utils.write_str(utils.TEMP_XML, str(soup))
        utils.open_code(utils.TEMP_XML)
        utils.open_url(x["meta.elem_url"])

    # img, img.newchar, span.idx_annotation01, span.idx_annotation03, span.idx_proofreading01, span.idx_proofreading02, span.idx_proofreading03,
    if 0:
        tags = list(soup.select("img"))
        tags
        [t.text for t in tags]

    # 제거
    tags = list(soup.select("페이지"))
    for t in tags:
        t.decompose()

    # 보존(unwrap)

    # 변환

    # Extracting and replacing NER info with unique identifiers
    tags1: list[Tag] = list(soup.find_all())
    tags = tags1
    tags = [t for t in tags if t.name.strip() not in {"강", "목"}]
    # leaf node만 선택!
    tags = [t for t in tags if not t.find()]
    ner_info: dict[str, dict[str, Any]] = {}
    if 0:
        t = random.choice(tags)
    for t in tags:
        uid1 = str(uuid.uuid4())
        t_class1 = t.name.strip()
        ner_info[uid1] = {"text": t.text.strip(), "label": t_class1}
        _ = t.replace_with(uid1)

    # body_ko를 위한 변환: br, p.paragraph 끝에 \n 추가
    if 0:
        tags = list(soup.select("br, p.paragraph"))
        for t in tags:
            t.append("\n")

    # body_ko
    body_text = utils.remove_whites(soup.text)

    # Replace UUID with XML-style NER
    if 0:
        unique_id, info = random.choice(list(ner_info.items()))
    for unique_id, info in ner_info.items():
        assert (
            unique_id in body_text
        ), f"unique_id not found: {dict(x)} \n{x['meta.url']}\n{x['meta.elem_id']}"
        name1 = str(info["label"])
        if name1.startswith("idx_"):
            name1 = name1[len("idx_") :]
        assert ">" not in name1, "bad ner name"
        text1 = info["text"]
        xml1 = f"<{_PREF}{name1}>{text1}</{_PREF}{name1}>"
        body_text = body_text.replace(unique_id, xml1)

    if 0:
        utils.write_str(utils.TEMP_HTML, str(soup))
        utils.open_file(utils.TEMP_HTML)
        utils.write_str(utils.TEMP_TXT, body_text)
        utils.open_file(utils.TEMP_TXT)

    return body_text


def gen_parse2_file() -> None:
    # read
    df = utils.read_df(sroot.CLEAN2_PQ)
    df.sample(1).iloc[0].to_dict()

    # sort by size
    if 0:
        df["size"] = df["meta.elem_body_text"].progress_apply(len)
        df["size"].describe()
        df.sort_values("size", inplace=True, ignore_index=True, ascending=False)
        df.drop(columns=["size"], inplace=True)

    # sample
    if 0:
        x = df[:100].sample(1).iloc[0].to_dict()  # large
        html = x["meta.elem_body_html"]
        ctool.find_xml_tags(html=html)
        utils.open_url(x["meta.elem_url"])
        #
        result1 = df["meta.elem_title_html"][:100].parallel_apply(ctool.find_xml_tags)
        result2 = df["meta.elem_body_html"][:500].parallel_apply(ctool.find_xml_tags)
        result: utils.SeriesType = pd.concat([result1, result2], ignore_index=True)
        found_tags = sorted(
            set([s.strip() for s in ";".join(result.unique()).split(";")])
        )
        found_tags = [t for t in found_tags if len(t) >= 1]
        logger.debug(", ".join(found_tags))
        # 강, 건물, 관서, 관직, 능원, 도삭, 목, 물건, 서명, 세주, 인물, 지명, 페이지
        if 0:
            sel1 = random.choice(found_tags)
        sel2url = {}
        for sel1 in tqdm(found_tags):
            sel2 = sel1.split(".")[-1]
            idx = df["meta.elem_body_html"].str.contains(sel2, regex=False)
            df1 = df[idx].reset_index(drop=True)
            n_safe = min(3, len(df1))
            urls = sorted(df1.sample(n_safe)["meta.url"].to_list())
            sel2url[sel2] = urls
        sel2url
        #
        x = df[-100:].sample(1).iloc[0].to_dict()  # small
        x = df[:100].sample(1).iloc[0].to_dict()  # large
        idx = df["meta.elem_id"] == "koa_10609016_001"
        idx = df["meta.elem_id"] == "kga_11207007_004"  # bad ner example
        # search
        idx = df["meta.elem_body_html"].str.contains("페이지", regex=False)
        x = df[idx].sample(1).iloc[0].to_dict()
        print(parse_elem_both_html(x=x))
        print(etool.xml2plaintext(parse_elem_both_html(x=x)))
        utils.open_url(x["meta.elem_url"])
        # random
        x = df.sample(1).iloc[0].to_dict()
        print(parse_elem_both_html(x=x))
        utils.open_url(x["meta.url"])

    # parse html
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    df["meta.elem_body_text"] = df.parallel_apply(  # type: ignore
        parse_elem_both_html, axis=1
    )
    df["meta.elem_title_text"] = df.parallel_apply(  # type: ignore
        lambda x: parse_elem_both_html(x=x, col="meta.elem_title_html"), axis=1
    )
    df = utils.replace_blank_to_none(df)

    # sort cols
    df = df[sorted(df.columns)].reset_index(drop=True)

    # sort rows
    kcols = ["meta.data_id"]
    logger.debug(df.groupby(kcols).size().value_counts())
    df.sort_values(kcols, inplace=True, ignore_index=True)

    # save
    utils.write_df2(sroot.PARSE2_PQ, df)


def main() -> None:
    # parse body html to text

    # init
    tqdm.pandas()
    pandarallel.initialize(progress_bar=True)
    #
    gen_parse2_file()  # 168.0M, 2e96927c


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
            # python -m src.crawl.drri_hj.parse2
            typer.run(main)

_ = {
    "강": [],
    "건물": [
        "https://kyudb.snu.ac.kr/series/directView.do?itemcd=ILS&bookcd=GK12811_00&pyear=1785&pmonth=09&pyun=0&pday=24",
        "https://kyudb.snu.ac.kr/series/directView.do?itemcd=ILS&bookcd=GK12811_00&pyear=1790&pmonth=10&pyun=0&pday=21",
        "https://kyudb.snu.ac.kr/series/directView.do?itemcd=ILS&bookcd=GK12814_00&pyear=1844&pmonth=06&pyun=0&pday=03",
    ],
    "관서": [
        "https://kyudb.snu.ac.kr/series/directView.do?itemcd=ILS&bookcd=GK12811_00&pyear=1777&pmonth=11&pyun=0&pday=25",
        "https://kyudb.snu.ac.kr/series/directView.do?itemcd=ILS&bookcd=GK12811_00&pyear=1778&pmonth=06&pyun=1&pday=16",
        "https://kyudb.snu.ac.kr/series/directView.do?itemcd=ILS&bookcd=GK12811_00&pyear=1786&pmonth=08&pyun=0&pday=26",
    ],
    "관직": [
        "https://kyudb.snu.ac.kr/series/directView.do?itemcd=ILS&bookcd=GK12811_00&pyear=1798&pmonth=06&pyun=0&pday=05",
        "https://kyudb.snu.ac.kr/series/directView.do?itemcd=ILS&bookcd=GK12813_00&pyear=1804&pmonth=01&pyun=0&pday=20",
        "https://kyudb.snu.ac.kr/series/directView.do?itemcd=ILS&bookcd=GK12813_00&pyear=1820&pmonth=02&pyun=0&pday=19",
    ],
    "능원": [
        "https://kyudb.snu.ac.kr/series/directView.do?itemcd=ILS&bookcd=GK12811_00&pyear=1787&pmonth=06&pyun=0&pday=05",
        "https://kyudb.snu.ac.kr/series/directView.do?itemcd=ILS&bookcd=GK12811_00&pyear=1791&pmonth=03&pyun=0&pday=09",
        "https://kyudb.snu.ac.kr/series/directView.do?itemcd=ILS&bookcd=GK12815_00&pyear=1851&pmonth=03&pyun=0&pday=16",
    ],
    "도삭": [
        "https://kyudb.snu.ac.kr/series/directView.do?itemcd=ILS&bookcd=GK12813_00&pyear=1817&pmonth=03&pyun=0&pday=25",
        "https://kyudb.snu.ac.kr/series/directView.do?itemcd=ILS&bookcd=GK12814_00&pyear=1836&pmonth=12&pyun=0&pday=24",
        "https://kyudb.snu.ac.kr/series/directView.do?itemcd=ILS&bookcd=GK12815_00&pyear=1861&pmonth=07&pyun=0&pday=10",
    ],
    "목": [
        "https://kyudb.snu.ac.kr/series/directView.do?itemcd=ILS&bookcd=GK12811_00&pyear=1793&pmonth=08&pyun=0&pday=06",
        "https://kyudb.snu.ac.kr/series/directView.do?itemcd=ILS&bookcd=GK12813_00&pyear=1810&pmonth=05&pyun=0&pday=01",
        "https://kyudb.snu.ac.kr/series/directView.do?itemcd=ILS&bookcd=GK12813_00&pyear=1822&pmonth=08&pyun=0&pday=07",
    ],
    "물건": [
        "https://kyudb.snu.ac.kr/series/directView.do?itemcd=ILS&bookcd=GK12811_00&pyear=1767&pmonth=05&pyun=0&pday=26",
        "https://kyudb.snu.ac.kr/series/directView.do?itemcd=ILS&bookcd=GK12813_00&pyear=1808&pmonth=02&pyun=0&pday=25",
        "https://kyudb.snu.ac.kr/series/directView.do?itemcd=ILS&bookcd=GK12815_00&pyear=1863&pmonth=12&pyun=0&pday=08",
    ],
    "서명": [
        "https://kyudb.snu.ac.kr/series/directView.do?itemcd=ILS&bookcd=GK12811_00&pyear=1784&pmonth=04&pyun=0&pday=30",
        "https://kyudb.snu.ac.kr/series/directView.do?itemcd=ILS&bookcd=GK12813_00&pyear=1807&pmonth=06&pyun=0&pday=07",
        "https://kyudb.snu.ac.kr/series/directView.do?itemcd=ILS&bookcd=GK12816_00&pyear=1867&pmonth=04&pyun=0&pday=24",
    ],
    "세주": [
        "https://kyudb.snu.ac.kr/series/directView.do?itemcd=ILS&bookcd=GK12813_00&pyear=1819&pmonth=12&pyun=0&pday=22",
        "https://kyudb.snu.ac.kr/series/directView.do?itemcd=ILS&bookcd=GK12815_00&pyear=1862&pmonth=07&pyun=0&pday=15",
        "https://kyudb.snu.ac.kr/series/directView.do?itemcd=ILS&bookcd=GK12816_00&pyear=1869&pmonth=01&pyun=0&pday=03",
    ],
    "인물": [
        "https://kyudb.snu.ac.kr/series/directView.do?itemcd=ILS&bookcd=GK12811_00&pyear=1786&pmonth=11&pyun=0&pday=17",
        "https://kyudb.snu.ac.kr/series/directView.do?itemcd=ILS&bookcd=GK12813_00&pyear=1802&pmonth=06&pyun=0&pday=08",
        "https://kyudb.snu.ac.kr/series/directView.do?itemcd=ILS&bookcd=GK12816_00&pyear=1870&pmonth=12&pyun=0&pday=22",
    ],
    "지명": [
        "https://kyudb.snu.ac.kr/series/directView.do?itemcd=ILS&bookcd=GK12814_00&pyear=1845&pmonth=09&pyun=0&pday=21",
        "https://kyudb.snu.ac.kr/series/directView.do?itemcd=ILS&bookcd=GK12815_00&pyear=1854&pmonth=07&pyun=1&pday=04",
        "https://kyudb.snu.ac.kr/series/directView.do?itemcd=ILS&bookcd=GK12816_00&pyear=1879&pmonth=02&pyun=0&pday=10",
    ],
    "페이지": [
        "https://kyudb.snu.ac.kr/series/directView.do?itemcd=ILS&bookcd=GK12811_00&pyear=1786&pmonth=12&pyun=0&pday=30",
        "https://kyudb.snu.ac.kr/series/directView.do?itemcd=ILS&bookcd=GK12811_00&pyear=1793&pmonth=06&pyun=0&pday=11",
        "https://kyudb.snu.ac.kr/series/directView.do?itemcd=ILS&bookcd=GK12813_00&pyear=1820&pmonth=11&pyun=0&pday=05",
    ],
}
