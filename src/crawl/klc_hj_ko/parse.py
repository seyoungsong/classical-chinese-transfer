import random
import shutil
import sys
from importlib import reload
from pathlib import Path
from typing import Any

import humanize
import pandas as pd
import psutil
import typer
from bs4 import BeautifulSoup
from loguru import logger
from rich import pretty
from tqdm import tqdm

import src.crawl.klc_hj_ko.root as sroot
from src import utils


def cmd_unzip_7z() -> None:
    sroot.CRAWL_DL_DIR.mkdir(exist_ok=True, parents=True)
    free_space = humanize.naturalsize(
        shutil.disk_usage(sroot.CRAWL_DL_DIR).free, gnu=True
    )
    cmd = f"""
    # FREE: {free_space}
    7za l {sroot.CRAWL_7Z}
    7za x {sroot.CRAWL_7Z} -o{sroot.CRAWL_DL_DIR} -y
    """
    sroot.SCRIPT_DIR.mkdir(exist_ok=True, parents=True)
    utils.write_sh(sroot.SCRIPT_DIR / "unzip_crawl_7z.sh", cmd)


def parse1(x1: dict[Any, Any]) -> list[dict[str, Any]]:
    _fname = x1["fname"]
    _data_id = x1["data_id"]
    _url_raw = f"""https://db.itkc.or.kr/dir/node?dataId={_data_id}&viewSync=OT&viewSync2=KP"""  # fmt: off
    _url_web = f"""https://db.itkc.or.kr/dir/item?itemId=BT#/dir/node?dataId={_data_id}&viewSync=OT&viewSync2=KP"""  # fmt: off

    if 0:
        utils.open_file(_fname)
        utils.open_code(_fname)
        utils.open_url(_url_raw)
        utils.open_url(_url_web)

    # read
    fname = Path(x1["fname"])
    html = fname.read_text(encoding="utf-8")
    soup = BeautifulSoup(html, "lxml")

    # check
    if len(html) < 100:
        raise ValueError("HTML too short")

    # check
    error_str = "시스템 오류"
    if error_str in soup.text.strip():
        raise ValueError("시스템 오류 in HTML")

    # page_path
    t = soup.select_one("div.path")
    assert t is not None, "page_path missing"
    page_path = utils.squeeze_whites(t.text.strip())

    # page_title
    # if missing, it is probably a higher level page with index of items.
    t = soup.select_one("h3.fl")
    assert t is not None, "page_title missing (safe)"
    page_title = utils.squeeze_whites(t.text)

    # parse
    elems = list(soup.select("div.gisa-wrap"))
    len(elems)
    data_list: list[dict[str, str]] = []
    if 0:
        elem = random.choice(elems)
    for elem_idx, elem in enumerate(elems):
        d1: dict[str, Any] = {}
        d1["x1"] = dict(x1)
        d1["elem_idx"] = elem_idx
        d1["page_path"] = page_path
        d1["page_title"] = page_title

        # data_id
        tags = list(elem.select("div"))
        tags = [t for t in tags if "data-viewnode-id" in t.attrs]
        # while not optimal, we drop data if not 1 to 1 correspondence.
        assert len(tags) > 0, "data_id missing (bad)"
        assert len(tags) == 1, "data_id not single (safe)"
        data_id = tags[0].attrs["data-viewnode-id"]
        item_id = data_id.split("_")[1]
        url = f"https://db.itkc.or.kr/dir/item?itemId={item_id}#/dir/node?dataId={data_id}"
        d1["data_id"] = data_id
        d1["url"] = url

        # title
        tags = list(elem.select("div.text_body_tit"))
        assert len(tags) == 1, "title not single"
        t = tags[0]
        d1["title_html"] = str(t)
        d1["title_text"] = utils.squeeze_whites(t.text)

        # dci
        tags = list(elem.select("div.dci-pane"))
        assert len(tags) == 1, "dci not single"
        t = tags[0]
        d1["dci_html"] = str(t)
        d1["dci_text"] = utils.squeeze_whites(t.text)

        # body
        tags = list(elem.select("div.text_body"))
        assert len(tags) == 1, "text_body not single"
        t = tags[0]
        d1["body_html"] = str(t)
        d1["body_text"] = utils.squeeze_whites(t.text)

        # copyright (opt)
        tags = list(elem.select("div.text_copyright"))
        assert len(tags) <= 1, "text_copyright too many"
        if len(tags) == 1:
            t = tags[0]
            d1["copyright_html"] = str(t)
            d1["copyright_text"] = utils.squeeze_whites(t.text)

        # dict
        data_list.append(d1)

    if len(data_list) == 0:
        # https://db.itkc.or.kr/dir/item?itemId=IT#/dir/node?dataId=ITKC_IT_U0_A39_12A_20A
        raise ValueError("data_list is empty")

    return data_list


def gen_parse_json1(x1: dict[Any, Any]) -> None:
    fname2 = Path(x1["fname2"])
    try:
        y1 = parse1(x1=x1)
    except Exception as e:
        if "safe" not in repr(e):
            logger.error(f"Exception: [ {repr(e)} ], x1: [ {dict(x1)} ]")
        y1 = [{"x1": dict(x1), "error": repr(e)}]
    fname2.parent.mkdir(parents=True, exist_ok=True)
    utils.write_json2(fname2, y1)


def gen_parse_dir() -> None:
    # init
    num_proc = psutil.cpu_count() - 1

    # get todo list
    ds = utils.read_ds(sroot.CRAWL_TASK_PQ)
    x1 = ds.shuffle()[0]
    assert Path(x1["fname"]).is_file(), "random test not work"

    # source fname: fix if needed
    if 0:
        ds = ds.map(
            lambda x1: {"fname": str(sroot.CRAWL_DL_DIR / Path(x1["fname"]).name)},
            num_proc=num_proc,
        )

    # target fname
    ds = ds.map(
        lambda x1: {
            "fname2": str(sroot.PARSE_DIR / Path(x1["fname"]).with_suffix(".json").name)
        },
        num_proc=num_proc,
    )
    ds.shuffle()[0]

    # sort by size
    ds = ds.map(
        lambda x1: {"size": Path(x1["fname"]).stat().st_size}, num_proc=num_proc
    )
    ds = ds.sort("size")

    # test
    if 0:
        x1 = ds[0]
        x1 = ds[1]
        x1 = ds[-1]
        x1 = ds[-2]
        x1 = ds.filter(
            lambda x: x["data_id"] == "ITKC_BT_1366A_0190_000_0210", num_proc=num_proc
        )[0]
        x1 = ds.shuffle()[0]
        parse1(x1=x1)
        gen_parse_json1(x1=x1)
        utils.open_file(x1["fname"])
        utils.open_file(x1["fname2"])

    # parse
    ds = ds.shuffle(seed=42)
    ds.map(
        gen_parse_json1, batched=False, load_from_cache_file=False, num_proc=num_proc
    )
    logger.success("gen_parse_dir done")


def gen_parse_file() -> None:
    # files
    fnames = sorted(sroot.PARSE_DIR.rglob("*.json"))
    len(fnames)  # 186373

    # read
    data = []
    fname1 = random.choice(fnames)
    for fname1 in tqdm(fnames):
        lx = utils.read_json2(fname1)
        data.extend(lx)

    # flatten
    df = pd.json_normalize(data)
    df.info()
    logger.debug(len(df))  # 347093

    # save
    sroot.PARSE_PQ.parent.mkdir(exist_ok=True, parents=True)
    utils.write_df2(sroot.PARSE_PQ, df)


def main() -> None:
    # parse raw files into jsonl (keep basic html and text)
    cmd_unzip_7z()
    gen_parse_dir()  # 2min for 186k files (m2 pro, 9 threads)
    gen_parse_file()  # 696.8M, e1daf15a


if __name__ == "__main__":
    if hasattr(sys, "ps1"):
        pretty.install()
        reload(utils)
        reload(sroot)
    else:
        with logger.catch(onerror=lambda _: sys.exit(1)):
            # python -m src.crawl.klc_hj_ko.parse
            typer.run(main)
