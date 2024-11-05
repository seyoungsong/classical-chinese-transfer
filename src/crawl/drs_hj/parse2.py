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

import src.crawl.drs_hj.root as sroot
import src.tool.corpus as ctool
import src.tool.eval as etool
from src import utils

_PREF = utils.NER_PREF


def parse_elem_body_html(x: dict[str, Any]) -> str:  # noqa: C901
    html: str = x["meta.elem_body_html"]
    soup = BeautifulSoup(html, "lxml")
    if 0:
        utils.write_str(utils.TEMP_HTML, str(soup))
        utils.open_file(utils.TEMP_HTML)
        utils.open_code(utils.TEMP_HTML)
        utils.open_url(x["meta.url"])

    # img, span.idx_annotation01, span.idx_book, span.idx_person, span.idx_place

    if 0:
        tags = list(soup.select("img"))
        tags
        [t.text for t in tags]
        [str(t) for t in tags]

    # 제거

    # 보존(unwrap)

    # 변환: img.newchar는 식별불가자.
    tags = list(soup.select("img"))
    tags = [t for t in tags if "newchar" in str(t)]
    for t in tags:
        _ = t.replace_with("▩")

    # Extracting and replacing NER info with unique identifiers
    _clas = "idx_book,idx_person,idx_place".split(",")
    _tags = "span,a".split(",")
    _sels = ", ".join([f"{s}.{c}" for s in _tags for c in _clas])
    tags = list(soup.select(_sels))
    # leaf node만 선택!
    tags = [t for t in tags if not t.find(_tags)]
    ner_info: dict[str, dict[str, Any]] = {}
    if 0:
        t = random.choice(tags)
    for t in tags:
        uid1 = str(uuid.uuid4())
        t_class = set(t.attrs["class"]).intersection(_clas)
        assert len(t_class) == 1, f"bad t_class: {t}"
        t_class1 = t_class.pop()
        ner_info[uid1] = {"text": t.text.strip(), "label": t_class1}
        _ = t.replace_with(uid1)

    # body_ko를 위한 변환: br, p.paragraph 끝에 \n 추가
    tags = list(soup.select("br, p.paragraph"))
    for t in tags:
        t.append("\n")

    # body_ko
    body_lines = [utils.squeeze_whites(s) for s in soup.text.strip().splitlines()]
    body_lines = [s for s in body_lines if s]
    body_text = "\n".join(body_lines)

    # Replace UUID with XML-style NER
    if 0:
        unique_id, info = random.choice(list(ner_info.items()))
    for unique_id, info in ner_info.items():
        assert (
            unique_id in body_text
        ), f"unique_id not found: {dict(x)} \n{x['meta.url']}\n{x['meta.data_id']}"
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
        ctool.find_special_tags(html=html)
        utils.open_url(x["meta.url"])
        #
        if 0:
            result1 = df["meta.elem_title_html"][:100].parallel_apply(
                ctool.find_special_tags
            )
            result1
        result2 = df["meta.elem_body_html"][:500].parallel_apply(
            ctool.find_special_tags
        )
        result: utils.SeriesType = pd.concat([result2], ignore_index=True)
        found_tags = sorted(
            set([s.strip() for s in ";".join(result.unique()).split(";")])
        )
        found_tags = [t for t in found_tags if len(t) >= 1]
        logger.debug(", ".join(found_tags))
        # img, span.idx_annotation01, span.idx_book, span.idx_person, span.idx_place
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
        idx = df["meta.data_id"] == "SJW-K11120040-02000"
        idx = df["meta.data_id"] == "kga_11207007_004"  # bad ner example
        # search
        idx = df["meta.elem_body_html"].str.contains("idx_proofreading03", regex=False)
        x = df[idx].sample(1).iloc[0].to_dict()
        print(parse_elem_body_html(x=x))
        print(etool.xml2plaintext(parse_elem_body_html(x=x)))
        utils.open_url(x["meta.url"])
        # random
        x = df.sample(1).iloc[0].to_dict()
        print(parse_elem_body_html(x=x))
        utils.open_url(x["meta.url"])

    # parse html
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    df["meta.elem_body_text"] = df.parallel_apply(  # type: ignore
        parse_elem_body_html, axis=1
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
    gen_parse2_file()  # 742.2M, 4b17a7a8


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
            # python -m src.crawl.drs_hj.parse2
            typer.run(main)

_ = {
    "img": [
        "https://sjw.history.go.kr/id/SJW-E02120190-01900",  # newchar
        "https://sjw.history.go.kr/id/SJW-H02010010-00800",
        "https://sjw.history.go.kr/id/SJW-K11120040-02000",
    ],
    "idx_annotation01": [
        "https://sjw.history.go.kr/id/SJW-B04030150-00600",  # 원주?
        "https://sjw.history.go.kr/id/SJW-F27090190-01800",
        "https://sjw.history.go.kr/id/SJW-K21070240-00700",
    ],
    "idx_book": [
        "https://sjw.history.go.kr/id/SJW-B02060040-00300",
        "https://sjw.history.go.kr/id/SJW-F14100090-01400",
        "https://sjw.history.go.kr/id/SJW-F30030110-02600",
    ],
    "idx_person": [
        "https://sjw.history.go.kr/id/SJW-G05080080-00800",
        "https://sjw.history.go.kr/id/SJW-G10110040-00400",
        "https://sjw.history.go.kr/id/SJW-J01060090-01200",
    ],
    "idx_place": [
        "https://sjw.history.go.kr/id/SJW-A01080080-00600",
        "https://sjw.history.go.kr/id/SJW-A24040060-00300",
        "https://sjw.history.go.kr/id/SJW-K28030220-02900",
    ],
}
