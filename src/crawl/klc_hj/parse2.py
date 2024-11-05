import random
import sys
import uuid
from importlib import reload
from typing import Any

import pandas as pd
import typer
from bs4 import BeautifulSoup
from loguru import logger
from pandarallel import pandarallel
from rich import pretty
from tqdm import tqdm

import src.crawl.klc_hj.root as sroot
import src.tool.corpus as ctool
from src import utils

_PREF = utils.NER_PREF


def parse_elem_body_html(x: dict[str, Any]) -> str:  # noqa: C901
    html: str = x["meta.elem_body_html"]
    soup = BeautifulSoup(html, "lxml")
    if 0:
        utils.write_str(utils.TEMP_HTML, str(soup))
        utils.open_file(utils.TEMP_HTML)
        utils.open_code(utils.TEMP_HTML)
        utils.open_url(x["meta.elem_url"])

    # tags: ["img", "span.jusok", "span.xsl_wonju", "span style="color:#5fb636""]
    # img, span.jusok, span.xsl_img_open, span.xsl_tbl_open, span.xsl_wonju, table

    # 보존: 원주, 하단주석링크
    tags = list(soup.select("span.jusok, span.xsl_wonju"))
    for t in tags:
        _ = t.unwrap()

    # 제거: 새창열기 버튼, 테이블, 이미지(팔괘?)
    tags = list(soup.select("span.xsl_img_open, span.xsl_tbl_open, table, img"))
    for t in tags:
        t.decompose()

    # 변환: br은 시의 new line.
    tags = list(soup.select("br"))
    for t in tags:
        _ = t.replace_with("\n" + t.text.strip())

    # Extracting and replacing NER info with unique identifiers
    tags = list(soup.select("span"))
    tags = [t for t in tags if "color" in t.attrs.get("style", "").lower()]
    ner_info: dict[str, dict[str, Any]] = {}
    for tag in tags:
        uid1 = str(uuid.uuid4())  # Generating a unique identifier
        ner_info[uid1] = {"text": tag.text.strip(), "style": tag.attrs["style"]}
        _ = tag.replace_with(uid1)

    # body_ko
    tags = list(
        soup.select("div.text_body div.xsl_para, div.text_body div.xsl_para_tit")
    )
    body_text = "\n".join([t.text.strip() for t in tags]).strip()

    # Replace UUID with XML-style NER
    if 0:
        unique_id, info = random.choice(list(ner_info.items()))
    for unique_id, info in ner_info.items():
        assert (
            unique_id in body_text
        ), f"unique_id not found: {dict(x)} \n{x['meta.url']}\n{x['meta.elem_id']}"
        name1 = str(info["style"])
        if name1.startswith("color:#"):
            name1 = name1[7:]
        assert ">" not in name1, "bad ner name"
        text1 = info["text"]
        xml1 = f"<{_PREF}{name1}>{text1}</{_PREF}{name1}>"
        body_text = body_text.replace(unique_id, xml1)

    if 0:
        utils.write_str(utils.TEMP_HTML, str(soup))
        utils.open_file(utils.TEMP_HTML)
        utils.write_str(utils.TEMP_TXT, body_text)
        utils.open_file(utils.TEMP_TXT)
        utils.temp_diff(body_text, body_text)

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
        ctool.find_special_tags(html=html)
        utils.open_url(x["meta.elem_url"])
        #
        result1 = df["meta.elem_title_html"][:100].parallel_apply(
            ctool.find_special_tags
        )
        result2 = df["meta.elem_body_html"][:100].parallel_apply(
            ctool.find_special_tags
        )
        result: utils.SeriesType = pd.concat([result1, result2], ignore_index=True)
        found_tags = sorted(
            set([s.strip() for s in ";".join(result.unique()).split(";")])
        )
        found_tags = [t for t in found_tags if len(t) >= 1]
        logger.debug(", ".join(found_tags))
        # img, span.jusok, span.xsl_img_open, span.xsl_wonju
        x = df[-100:].sample(1).iloc[0].to_dict()  # small
        x = df[:100].sample(1).iloc[0].to_dict()  # large
        #
        idx = df["meta.elem_id"] == "ITKC_MP_0184A_0030_010_0010"
        idx = df["meta.elem_id_type"].isin(["MP", "KP"]) & df[
            "meta.elem_body_html"
        ].str.contains("color:")
        idx = df["meta.elem_id_type"].isin(["MP", "KP"]) & df[
            "meta.elem_body_html"
        ].str.contains("jusok")
        idx = df["meta.elem_body_html"].str.contains("color:")
        #
        x = df[idx].sample(1).iloc[0].to_dict()
        x = df.sample(1).iloc[0].to_dict()  # random
        print(parse_elem_body_html(x=x))
        utils.open_url(x["meta.elem_url"])

    # parse html
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    df["meta.elem_body_text"] = df.parallel_apply(  # type: ignore
        parse_elem_body_html, axis=1
    )
    df = utils.replace_blank_to_none(df)

    # sort cols
    df = df[sorted(df.columns)].reset_index(drop=True)

    # sort rows
    kcols = ["meta.data_id", "meta.elem_id"]
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
    gen_parse2_file()  # 809.0M, 3007f39b


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
            # python -m src.crawl.klc_hj.parse2
            typer.run(main)
