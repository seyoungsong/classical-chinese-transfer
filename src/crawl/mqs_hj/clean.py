import re
import sys
from importlib import reload

import hanja
import typer
from bs4 import BeautifulSoup
from loguru import logger
from pandarallel import pandarallel
from rich import pretty
from tqdm import tqdm

import src.crawl.mqs_hj.root as sroot
from src import utils


def parse_body_hj_html(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")

    # idx_annotation01, idx_annotation02: 원문에서 글자 정정한 걸 표시한 듯. 보존.
    # https://sillok.history.go.kr/mc/id/msilok_004_0630_0010_0010_0080_0010

    # idx_annotation03: 원본에 없는 단어 풀이 주석
    # https://sillok.history.go.kr/mc/id/msilok_009_1120_0010_0010_0010_0020
    tags = list(soup.select("span.idx_annotation03"))
    for t in tags:
        _ = t.replace_with("")

    # idx_annotation04: inline 정정 내역 주석인 듯
    # https://sillok.history.go.kr/mc/id/msilok_013_0640_0010_0010_0040_0010
    tags = list(soup.select("span.idx_annotation04"))
    for t in tags:
        _ = t.replace_with("")

    # idx_annotation05: 眉批(미비) 책·서류 등의 윗부분에 써넣는 평어(評語)나 주석. 원문에 없으니 삭제.
    # https://sillok.history.go.kr/mc/id/msilok_014_0040_0040_0050_0190_0010
    tags = list(soup.select("span.idx_annotation05"))
    for t in tags:
        _ = t.replace_with("")

    # idx_annotation06: 摺包(접포) 원문에는 없고, 구절의 끝을 나타내는 듯. 원문에 없으니 삭제.
    # https://sillok.history.go.kr/mc/id/qsilok_012_5160_0010_0010_0070_0020
    tags = list(soup.select("span.idx_annotation06"))
    for t in tags:
        _ = t.replace_with("")

    # text
    tags = list(soup.select("p.paragraph"))
    texts = [utils.squeeze_whites(t.text) for t in tags]
    s = "\n\n".join(texts)

    if 0:
        utils.write_str(utils.TEMP_HTML, str(soup))
        utils.open_file(utils.TEMP_HTML)
        utils.write_str(utils.TEMP_TXT, s)
        utils.open_file(utils.TEMP_TXT)

    return s


def parse_date_hj(date_hj: str) -> str:
    date_hj = str(
        hanja.translate(text=date_hj, mode="combination-text-reversed")
    ).strip()
    return date_hj


def get_date_id_ko(date_ko: str) -> str:
    # date_ko = '1665년 4월 19일'

    # strip whitespace
    date2 = utils.remove_whites(date_ko)
    # regex match
    pstr = r"^(?P<year>\d+)년(?P<yun>윤?)(?P<month>\d+)월(?P<day>\d+)일"
    p = re.compile(pstr)
    m = p.search(date2)
    assert m is not None, f'date_ko="{date_ko}"'
    d = m.groupdict()
    # year
    year = int(d["year"])
    # month
    month = int(d["month"])
    # yun_month (order: 5월, 윤5월, 6월)
    is_yun_month: bool = len(d["yun"]) == 1
    yun_month = int(is_yun_month)
    # day
    day_clean = str(d["day"]).replace("일", "")
    day = int(day_clean) if day_clean.isdigit() else "??"
    # date num
    date_id = f"{year:04}-{month:02}-{yun_month:01}-{day:02}"
    assert len(date_id) == 12
    return date_id


def gen_mqs_hj_clean_pkl() -> None:
    # read
    df = utils.read_df(sroot.MQS_HJ_SRC_PKL)
    df.sample(1).iloc[0].to_dict()
    df.info()
    df.columns

    # init
    tqdm.pandas()
    pandarallel.initialize(progress_bar=True)

    # cols
    cols = [
        "data_id",
        "date_hj",
        "body_hj",
        "body_hj_html",
        "error",
        "url",
    ]
    assert set(cols) == set(df.columns)
    df = df[cols].reset_index(drop=True)

    # drop error
    if 0:
        df[df["error"].notnull()].iloc[0].to_dict()
    df = df[df["error"].isnull()].reset_index(drop=True)
    df.drop(columns=["error"], inplace=True)

    # sort
    df["size"] = df["body_hj"].apply(lambda x: len(str(x)))
    df.sort_values("size", inplace=True, ignore_index=True, ascending=False)

    # sample
    if 0:
        x = df.sample(1).iloc[0].to_dict()  # random
        x = df.iloc[-1].to_dict()  # small
        x = df.iloc[0].to_dict()  # large
        idx = df["body_hj_html"].str.contains("idx_annotation05")
        x = df[idx].sample(1).iloc[0].to_dict()
        x
        #
        html = x["body_hj_html"]
        parse_body_hj_html(html)

    # parse html
    df.sort_values("data_id", inplace=True, ignore_index=True)
    df["body_hj"] = df["body_hj_html"].parallel_apply(parse_body_hj_html)

    # date_hj2
    df["date_hj2"] = df["date_hj"].parallel_apply(parse_date_hj)

    # date_ko (temp)
    day_map = utils.read_json(sroot.RESULT_DIR / "mqs-hj-day.json")
    df["temp_id"] = df["data_id"].apply(lambda x: str(x).rsplit("_", 1)[0])
    df["date_ko"] = df["temp_id"].progress_apply(lambda x: day_map.get(x, ""))
    assert (df["date_ko"] != "").all()

    # date_id
    df["date_id"] = df["date_ko"].progress_apply(get_date_id_ko)
    df.drop(columns=["date_ko"], inplace=True)

    # drop useless columns
    cols = [str(c) for c in df.columns if "html" in c or "size" in c or "temp" in c]
    df.drop(columns=cols, inplace=True)

    # uid
    df["uid"] = "mqs|id:" + df["data_id"] + "|date:" + df["date_id"]

    # url
    df["url"] = df["url"].apply(
        lambda x: f" {str(x).strip()} " if str(x) != "nan" else ""
    )

    # reorder columns
    df.columns
    cols = [
        "uid",
        "data_id",
        "date_id",
        "date_hj2",
        "date_hj",
        "body_hj",
        "url",
    ]
    assert set(cols) == set(df.columns)
    df = df[cols].reset_index(drop=True)

    # sample
    df.sample(1).iloc[0].to_dict()
    utils.sample_df(df)

    # save
    df.sort_values("uid", inplace=True, ignore_index=True)
    utils.write_df(sroot.MQS_HJ_CLEAN_PKL, df)  # 75.0M, 62a221bf
    utils.log_written(sroot.MQS_HJ_CLEAN_PKL)
    if 0:
        df = utils.read_df(sroot.MQS_HJ_CLEAN_PKL)
        utils.sample_df(df)


def main() -> None:
    gen_mqs_hj_clean_pkl()


if __name__ == "__main__":
    if hasattr(sys, "ps1"):
        pretty.install()
        reload(utils)
    else:
        with logger.catch(onerror=lambda _: sys.exit(1)):
            # python -m src.crawl.qing_hj.mqs_hj_clean
            typer.run(main)
