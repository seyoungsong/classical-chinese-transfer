import random
import re
import sys
from importlib import reload
from pathlib import Path

import pandas as pd
import typer
from bs4 import BeautifulSoup
from loguru import logger
from rich import pretty
from tqdm import tqdm

import src.crawl.mqs_hj.root as sroot
from src import utils


def gen_king_json() -> None:
    url = "https://sillok.history.go.kr/mc/inspectionList.do"
    html = utils.requests_get(url)
    soup = BeautifulSoup(html, "lxml")
    if 0:
        utils.write_str(utils.TEMP_HTML, str(soup))
        utils.open_file(utils.TEMP_HTML)
        utils.open_code(utils.TEMP_HTML)
    tags = list(soup.select("td > a"))
    king_map = {t.attrs["href"].split("'")[1]: t.text.strip() for t in tags}
    king_map = utils.sort_dict(king_map)
    assert len(king_map) == 28
    sroot.RESULT_DIR.mkdir(exist_ok=True)
    utils.write_json(sroot.RESULT_DIR / "mqs-hj-king.json", king_map)


def get_tree_type(king_id: str) -> str:
    if king_id.lower().startswith("m"):
        return "M"
    elif king_id.lower().startswith("q"):
        return "C"
    else:
        raise ValueError(f"king_id: {king_id}")


def gen_day_json() -> None:
    # doc_id_month. we ignore 총서, 부록
    king_map = utils.read_json(sroot.RESULT_DIR / "mqs-hj-king.json")
    day_map = {}
    king_list = sorted(list(king_map.keys()))
    king_list = utils.shuffle_list(king_list, seed=42)
    for king_id in tqdm(king_map.keys()):
        # https://sillok.history.go.kr/mc/ajaxExpandTree.do?id=msilok_007&level=5&treeType=M
        tree_type = get_tree_type(king_id)
        url = f"https://sillok.history.go.kr/mc/ajaxExpandTree.do?id={king_id}&level=5&treeType={tree_type}"
        html = utils.requests_get(url)
        soup = BeautifulSoup(html, "lxml")
        tags = list(soup.select("a"))
        tags = [t for t in tags if "년" in t.text]
        day_map1 = {t.attrs["href"].split("'")[1]: t.text.strip() for t in tags}
        day_map.update(day_map1)
    day_map = utils.sort_dict(day_map)
    utils.write_json(sroot.RESULT_DIR / "mqs-hj-day.json", day_map)
    assert len(day_map) == 164927
    if 0:
        day_map = utils.read_json(sroot.RESULT_DIR / "mqs-hj-day.json")


def write_day1_html(day1_id: str) -> None:
    # https://sillok.history.go.kr/mc/inspectionDayList.do?id=qsilok_007_0071_0010_0010_0020&treeType=C&dateInfo=1738년+6월+17일
    # https://sillok.history.go.kr/mc/id/qsilok_007_0989_0010_0010_0100_0010
    url = f"https://sillok.history.go.kr/mc/id/{day1_id}"
    try:
        html = utils.requests_get(url).strip()
        fname = sroot.TEMP_MQS_HJ_DAY_DIR / f"{day1_id}.html"
        fname.write_text(html, encoding="utf-8")
    except Exception as e:
        logger.error(f"Exception: {repr(e)} | url: {url}")


def write_day_dir() -> None:
    day_map = utils.read_json(sroot.RESULT_DIR / "mqs-hj-day.json")
    tqdm.pandas()
    df = pd.DataFrame([{"id": k, "date": v} for k, v in day_map.items()])
    df["date"] = df["date"].progress_apply(utils.squeeze_whites)
    df["id"].str.len().value_counts()
    idx1 = df["date"].apply(
        lambda x: "년" in x and "월" in x and "일" in x and "미상" not in x
    )
    idx2 = df["id"].str.len() == 30
    idx = idx1 & idx2
    df = df[idx].reset_index(drop=True)
    # https://sillok.history.go.kr/mc/inspectionDayList.do?id=qsilok_007_0071_0010_0010_0020&treeType=C&dateInfo=1738년+6월+17일
    df["url"] = df.progress_apply(  # type: ignore
        lambda x: f"https://sillok.history.go.kr/mc/inspectionDayList.do?id={x['id']}&treeType={get_tree_type(x['id'])}&dateInfo={str(x['date']).replace(' ', '+')}",
        axis=1,
    )
    assert len(df) == 164756
    assert df["id"].is_unique

    df["day1_id"] = df["id"] + "_0010"
    df["day1_url"] = df["day1_id"].apply(
        lambda x: f"https://sillok.history.go.kr/mc/id/{x}"
    )

    df.sample(1).iloc[0].to_dict()

    day1_ids = df["day1_id"].to_list()
    day1_ids = sorted(list(set(day1_ids)))
    assert len(day1_ids) == 164756

    # existing
    sroot.TEMP_MQS_HJ_DAY_DIR.mkdir(parents=True, exist_ok=True)
    done = [p.stem for p in list(sroot.TEMP_MQS_HJ_DAY_DIR.rglob("*.html"))]
    logger.debug(f"done={len(done)}")
    todo = sorted(list(set(day1_ids) - set(done)))
    todo = utils.shuffle_list(todo, seed=42)
    logger.debug(f"todo={len(todo)}")

    if 0:
        day1_id = todo[0]
        write_day1_html(day1_id)

    output = utils.pool_map(func=write_day1_html, xs=todo)
    logger.debug(f"len(output)={len(output)}")


def read_day1_html(fname: str) -> list[str]:
    p = Path(fname)
    url = f"https://sillok.history.go.kr/mc/id/{p.stem}"
    try:
        return read_day1_html_helper(fname)
    except Exception as e:
        logger.error(f"Exception: {repr(e)} | fname: {fname} | url: {url}")
        raise e


def read_day1_html_helper(fname: str) -> list[str]:
    p = Path(fname)
    url = f"https://sillok.history.go.kr/mc/id/{p.stem}"
    html = p.read_text()
    soup = BeautifulSoup(html, "lxml")

    if "해당하는 데이터가 없습니다." in soup.text:
        logger.warning(f'해당하는 데이터가 없습니다. | fname="{p}" | url: {url}')
        return [p.stem, "해당하는 데이터가 없습니다."]

    t = soup.select_one("a.btn_error")
    assert t is not None
    texts = t.attrs["href"].split("'")

    if "silok" not in str(t):
        logger.error(f'silok_id not found | fname="{p}" | url: {url}')
        raise RuntimeError(f'silok_id not found | fname="{p}" | url: {url}')
    day1_id = [s for s in texts if "silok" in s][0]

    if "기사" not in str(t):
        return [day1_id, "기사 not found"]

    gisa_str = [s for s in texts if "기사" in s][0]
    pattern = re.compile(r"/\s*?(?P<num>\d+)\s*?기사")
    m = pattern.search(gisa_str)
    assert m is not None
    max_num = int(m.groupdict()["num"])

    return [day1_id, str(max_num)]


def gen_mqs_hj_entry_pkl() -> None:
    fnames = [str(p) for p in list(sroot.TEMP_MQS_HJ_DAY_DIR.rglob("*.html"))]
    fnames = sorted(fnames)
    fnames = utils.shuffle_list(fnames, seed=42)
    if 0:
        fname = random.choice(fnames)
        result = read_day1_html(fname)
        print(result)
    if 0:
        for fname in tqdm(fnames):
            result = read_day1_html(fname)

    results = utils.pool_map(func=read_day1_html, xs=fnames)
    results_data = [{"id": s1, "num": s2} for s1, s2 in results]

    # save pkl
    df1 = pd.DataFrame(results_data)
    df1.drop_duplicates(subset="id", inplace=True, ignore_index=True)
    df1.sort_values("id", inplace=True, ignore_index=True)
    utils.write_df(sroot.MQS_HJ_ID_PKL, df1)  # 312.1K
    utils.log_written(sroot.MQS_HJ_ID_PKL)  # 2f4f9f99
    logger.debug(f"n={len(df1)}")

    # json sample
    safe_n = min(1000, len(df1))
    id_sample = {
        k: v for k, v in df1.sample(n=safe_n, random_state=42).sort_index().values
    }
    utils.write_json(sroot.RESULT_DIR / "mqs-hj-id-sample.json", id_sample)

    # parse
    df1 = utils.read_df(sroot.MQS_HJ_ID_PKL)
    #
    idx = df1["num"].str.contains("없습니다")
    df1 = df1[~idx].reset_index(drop=True)
    #
    idx = df1["num"].str.contains("기사 not found")
    df1.loc[idx, "num"] = "1"
    #
    df1["num"] = df1["num"].apply(int)
    #
    df1.sort_values("id", inplace=True, ignore_index=True)
    df1["stem"] = df1["id"].apply(lambda x: str(x).rsplit("_", 1)[0])
    #
    id_ll = [
        [f"{stem1}_{i:03}0" for i in range(1, n1 + 1)] for id1, n1, stem1 in df1.values
    ]
    id_list = utils.flatten(id_ll)
    #
    df2 = pd.DataFrame({"id": id_list})
    df2.sort_values("id", inplace=True, ignore_index=True)
    df2.drop_duplicates(subset="id", inplace=True, ignore_index=True)
    #
    utils.write_df(sroot.MQS_HJ_ENTRY_PKL, df2)  # 862.0K
    utils.log_written(sroot.MQS_HJ_ENTRY_PKL)  # 13127cc4
    logger.debug(f"n={len(df2)}")


def write_target_html(target_id: str) -> None:
    # https://sillok.history.go.kr/mc/inspectionDayList.do?id=qsilok_007_0071_0010_0010_0020&treeType=C&dateInfo=1738년+6월+17일
    # https://sillok.history.go.kr/mc/id/qsilok_007_0989_0010_0010_0100_0010
    url = f"https://sillok.history.go.kr/mc/id/{target_id}"
    try:
        html = utils.requests_get(url).strip()
        fname = sroot.TEMP_MQS_HJ_HTML_DIR / f"{target_id}.html"
        fname.write_text(html, encoding="utf-8")
    except Exception as e:
        logger.error(f"Exception: {repr(e)} | url: {url}")


def write_target_html_dir() -> None:
    # targets
    df = utils.read_df(sroot.MQS_HJ_ENTRY_PKL)
    target: list[str] = df["id"].to_list()
    target = sorted(list(set(target)))
    assert len(target) == 490306
    logger.debug(f"target={len(target)}")

    # existing
    sroot.TEMP_MQS_HJ_HTML_DIR.mkdir(parents=True, exist_ok=True)
    done = [p.stem for p in list(sroot.TEMP_MQS_HJ_HTML_DIR.rglob("*.html"))]
    logger.debug(f"done={len(done)}")

    # todo
    todo = sorted(list(set(target) - set(done)))
    todo = utils.shuffle_list(todo, seed=42)
    logger.debug(f"todo={len(todo)}")

    if 0:
        target_id = todo[0]
        write_target_html(target_id)

    output = utils.pool_map(func=write_target_html, xs=todo)
    logger.debug(f"output={len(output)}")


def main() -> None:
    if 0:
        gen_king_json()
        gen_day_json()
        write_day_dir()  # 43 min
        utils.subprocess_run(f"du -hd0 {sroot.TEMP_MQS_HJ_DAY_DIR}")  # 5.1G
        gen_mqs_hj_entry_pkl()
        write_target_html_dir()  # 3h?
        utils.subprocess_run(f"du -hd0 {sroot.TEMP_MQS_HJ_HTML_DIR}")  # 5.1G
    write_target_html_dir()


if __name__ == "__main__":
    if hasattr(sys, "ps1"):
        pretty.install()
        reload(utils)
    else:
        with logger.catch(onerror=lambda _: sys.exit(1)):
            # python -m src.crawl.mqs_hj
            typer.run(main)
