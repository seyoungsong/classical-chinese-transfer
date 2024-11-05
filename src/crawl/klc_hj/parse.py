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

import src.crawl.klc_hj.root as sroot
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


def cmd_fix_unzip_dir() -> None:
    target_dir = sroot.CRAWL_DL_DIR
    nested_dir = target_dir / target_dir.name
    if nested_dir.is_dir():
        logger.warning(f"nested_dir found: {nested_dir}")
    temp_dir = target_dir.parent / f"{target_dir.name}_temp"
    cmd = f"""
    mv {nested_dir} {temp_dir}
    du -hd0 {target_dir}
    rm -rf {target_dir}
    mv {temp_dir} {target_dir}
    """
    utils.write_sh(sroot.SCRIPT_DIR / "fix_unzip_dir.sh", cmd)


def parse1(x1: dict[Any, Any]) -> list[dict[str, Any]]:
    _fname = x1["fname"]
    _data_id = x1["data_id"]
    _url_raw = f"""https://db.itkc.or.kr/dir/node?dataId={_data_id}&viewSync=KP&viewSync2=TR"""  # fmt: off
    _url_web = f"""https://db.itkc.or.kr/dir/item?itemId=MO#/dir/node?dataId={_data_id}&viewSync=KP&viewSync2=TR"""  # fmt: off

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

    # check
    must_str = "data-viewnode-id"
    if must_str not in str(soup).lower():
        raise ValueError(f"{must_str} not found (safe)")

    # page_path
    t = soup.select_one("div.path")
    assert t is not None, "page_path missing"
    page_path = utils.squeeze_whites(t.text.strip())

    # page_title
    # if missing, it is probably a higher level page with index of items.
    tags = list(soup.select("h3"))
    assert len(tags) <= 1, "page_title too many"
    #
    t = soup.select_one("h3")
    assert t is not None, "page_title missing (safe)"
    page_title = utils.squeeze_whites(t.text)

    # parse
    tags = list(soup.select("div, section"))
    elems = [t for t in tags if "data-viewnode-id" in t.attrs]
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
        if "data-viewnode-id" in elem.attrs:
            data_id = elem.attrs["data-viewnode-id"]
        else:
            tags = list(elem.select("div"))
            tags = [t for t in tags if "data-viewnode-id" in t.attrs]
            # while not optimal, we drop data if not 1 to 1 correspondence.
            assert len(tags) > 0, "data_id missing (bad)"
            assert len(tags) == 1, "data_id not single (safe)"
            data_id = tags[0].attrs["data-viewnode-id"]

        # url
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
    # 나중에 jsonl로 바꾸자. 쉽게 파일을 append할 수 있음.
    utils.write_json2(fname2, y1)


def gen_parse_dir() -> None:
    # init
    num_proc = psutil.cpu_count() - 1

    # get todo list
    ds = utils.read_ds(sroot.CRAWL_TASK_PQ)
    x1 = ds.shuffle()[0]

    # source fname: fix if needed
    if not Path(x1["fname"]).parent.is_dir():
        ds = ds.map(
            lambda x1: {"fname": str(sroot.CRAWL_DL_DIR / Path(x1["fname"]).name)},
            num_proc=num_proc,
        )

    # check
    x1 = ds.shuffle()[0]
    assert Path(x1["fname"]).is_file(), f"File missing: {x1['fname']}"

    # gen target fname
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
        # get
        x1 = ds.filter(
            lambda x: x["data_id"] == "ITKC_MO_1033A_0060", num_proc=num_proc
        )[0]
        #
        x1 = ds[random.choice(range(100))]  # small
        x1 = ds[random.choice(range(len(ds) - 100, len(ds)))]  # large
        x1 = ds.shuffle()[0]  # random
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
    logger.debug(len(fnames))  # 655544

    # Step 2: Read and flatten each JSON file
    df_list = []  # to store the dataframes temporarily
    random.shuffle(fnames)
    for fname1 in tqdm(fnames):
        d1 = utils.read_json2(fname1)
        df1 = pd.json_normalize(d1)
        df_list.append(df1)
        del d1
    df = pd.concat(df_list, ignore_index=True)
    del df_list

    # flatten
    df.info()

    # save
    sroot.PARSE_PQ.parent.mkdir(exist_ok=True, parents=True)
    utils.write_df2(sroot.PARSE_PQ, df)
    logger.debug(len(df))


def main() -> None:
    # basic parse raw html files into jsonl
    tqdm.pandas()
    if 0:
        cmd_unzip_7z()
        cmd_fix_unzip_dir()
        gen_parse_dir()  # 6min for 656k files (m2 pro, 9 threads)
        gen_parse_file()  # 971.2M, 450cd703, 700632
    gen_parse_file()


if __name__ == "__main__":
    if hasattr(sys, "ps1"):
        pretty.install()
        reload(utils)
        reload(sroot)
    else:
        with logger.catch(onerror=lambda _: sys.exit(1)):
            # python -m src.crawl.klc_hj.parse
            typer.run(main)
