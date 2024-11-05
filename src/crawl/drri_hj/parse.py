import json
import random
import shutil
import sys
import urllib.parse
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

import src.crawl.drri_hj.root as sroot
from src import utils


def gen_url(x1: dict) -> str:  # type: ignore
    # convert
    x2 = {k: urllib.parse.quote(str(v)) for k, v in x1.items()}
    # https://kyudb.snu.ac.kr/series/rightItemView.do?nCnt=1000&pid=1000&py_king_nm=%EC%A0%95%EC%A1%B0&book_cd=GK12811_00&id=1000&nlevel=4&py_year=1780&py_month=06&py_yun=0&py_day=16&py_century=&vol_no=0088&div_menu=1&item_cd=ILS&cate=1&upd_yn=&searchString=&kwd=
    # https://kyudb.snu.ac.kr/series/rightItemView.do?nCnt=1000&pid=1000&py_king_nm=정조&book_cd=GK12811_00&id=1000&nlevel=4&py_year=1780&py_month=06&py_yun=0&py_day=16&py_century=&vol_no=0088&div_menu=1&item_cd=ILS&cate=1&upd_yn=&searchString=&kwd=
    # https://kyudb.snu.ac.kr/series/directView.do?itemcd=ILS&bookcd=GK12813_00&pyear=1801&pmonth=01&pyun=0&pday=08
    url = f"https://kyudb.snu.ac.kr/series/directView.do?itemcd=ILS&bookcd={x2['book_cd']}&pyear={x2['py_year']}&pmonth={x2['py_month']}&pyun={x2['py_yun']}&pday={x2['py_day']}"

    return url


def cmd_unzip_7z() -> None:
    sroot.LV4_DL_DIR.mkdir(exist_ok=True, parents=True)
    free_space = humanize.naturalsize(
        shutil.disk_usage(sroot.LV4_DL_DIR).free, gnu=True
    )
    cmd = f"""
    # FREE: {free_space}
    7za l {sroot.LV4_7Z}
    7za x {sroot.LV4_7Z} -o{sroot.LV4_DL_DIR} -y
    """
    sroot.SCRIPT_DIR.mkdir(exist_ok=True, parents=True)
    utils.write_sh(sroot.SCRIPT_DIR / "unzip_lv4_7z.sh", cmd)


def cmd_fix_unzip_dir() -> None:
    target_dir = sroot.LV4_DL_DIR
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


def parse1(x1: dict[Any, Any]) -> list[dict[str, Any]]:  # noqa: C901
    if 0:
        _fname = x1["fname"]
        _url_data = x1["url"]  # fmt: off
        _url_web = gen_url(x1)
        utils.open_file(_fname)
        utils.open_code(_fname)
        utils.open_url(_url_data)
        utils.open_url(_url_web)

    # read
    fname = Path(x1["fname"])
    json_str = fname.read_text(encoding="utf-8")

    # check
    if len(json_str) < 100:
        raise ValueError("HTML too short")

    # check
    must_str = "itemList".lower()
    if must_str not in str(json_str).lower():
        raise ValueError(f"{must_str} not found (safe)")

    # parse
    j1 = json.loads(json_str)
    assert isinstance(j1, dict), "d1 not dict"
    j2: dict[str, Any] = utils.notnull_collection(j1)  # type: ignore
    if 0:
        utils.write_json(utils.TEMP_JSON, j2)
        utils.open_code(utils.TEMP_JSON)

    # page_date
    html = j2["bodyTop"]
    soup = BeautifulSoup(html, "lxml")
    page_date = utils.squeeze_whites(soup.text.strip())

    # url2
    url2 = gen_url(x1)

    # parse
    items = j2["itemList"]
    len(items)
    data_list: list[dict[str, str]] = []
    if 0:
        item = items[0]
        item = random.choice(items)
        utils.squeeze_whites(item)
    for item_idx, item in enumerate(items):
        # parse
        html = item["tbl_conts_ori"]
        soup = BeautifulSoup(html, "xml")
        if 0:
            utils.write_str(utils.TEMP_XML, str(soup))
            utils.open_code(utils.TEMP_XML)
            utils.open_url(url2)

        # elems
        elems = list(soup.select("기사"))
        if 0:
            elem = elems[0]
            elem = random.choice(elems)
            utils.squeeze_whites(elem.text)
        for elem_idx, elem in enumerate(elems):
            d1: dict[str, Any] = {}
            d1["x1"] = dict(x1)
            d1["item_idx"] = item_idx
            d1["elem_idx"] = elem_idx
            d1["page_date"] = page_date
            d1["url2"] = url2

            # title (강)
            tags = list(elem.select("강"))
            assert len(tags) <= 1, "title not 0 or 1"
            if len(tags) == 1:
                t = tags[0]
                d1["title_html"] = str(t)
                d1["title_text"] = utils.squeeze_whites(t.text)

            # body
            tags = list(elem.select("목"))
            assert len(tags) <= 1, "body not 0 or 1"
            if len(tags) == 1:
                t = tags[0]
                d1["body_html"] = str(t)
                d1["body_text"] = utils.squeeze_whites(t.text)

            # etc
            tags = list(elem.select("강, 목"))
            for t in tags:
                t.decompose()
            etc_text = utils.squeeze_whites(elem.text)
            if len(etc_text) > 0:
                d1["etc_html"] = str(elem)
                d1["etc_text"] = etc_text

            # dict
            data_list.append(d1)

    if len(data_list) == 0:
        # https://db.itkc.or.kr/dir/item?itemId=IT#/dir/node?dataId=ITKC_IT_U0_A39_12A_20A
        raise ValueError("data_list is empty")

    return data_list


def gen_parse_jsonl1(x1: dict[Any, Any]) -> None:
    fname2 = Path(x1["fname2"])
    try:
        y1 = parse1(x1=x1)
    except Exception as e:
        if "safe" not in repr(e):
            logger.error(f"Exception: [ {repr(e)} ], x1: [ {dict(x1)} ]")
        y1 = [{"x1": dict(x1), "error": repr(e)}]
    fname2.parent.mkdir(parents=True, exist_ok=True)
    # jsonl for future concat
    df1 = pd.json_normalize(y1)
    df1.to_json(fname2, orient="records", lines=True, force_ascii=False)
    del df1


def gen_parse_dir() -> None:
    # init
    num_proc = psutil.cpu_count() - 1

    # get todo list
    ds = utils.read_ds(sroot.LV4_TASK_PQ)
    x1 = ds.shuffle()[0]

    # source fname: fix if needed
    if not Path(x1["fname"]).parent.is_dir():
        ds = ds.map(
            lambda x1: {"fname": str(sroot.LV4_DL_DIR / Path(x1["fname"]).name)},
            num_proc=num_proc,
        )

    # check
    x1 = ds.shuffle()[0]
    assert Path(x1["fname"]).is_file(), f"File missing: {x1['fname']}"

    # ignore missing
    ds = ds.filter(lambda x1: Path(x1["fname"]).is_file(), num_proc=num_proc)
    logger.debug(ds)

    # gen target fname
    ds = ds.map(
        lambda x1: {
            "fname2": str(
                sroot.PARSE_DIR / Path(x1["fname"]).with_suffix(".jsonl").name
            )
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
        x1 = ds.filter(lambda x: x["temp_id"] == "L33511", num_proc=num_proc)[0]
        #
        x1 = ds[random.choice(range(100))]  # small
        x1 = ds[random.choice(range(len(ds) - 100, len(ds)))]  # large
        #
        x1 = ds.shuffle()[0]  # random
        parse1(x1=x1)
        gen_parse_jsonl1(x1=x1)
        utils.open_file(x1["fname"])
        utils.open_code(x1["fname2"])
        #
        x1 = ds.shuffle()[0]  # random
        y1 = pd.json_normalize(parse1(x1)).to_dict(orient="records")
        utils.write_json(utils.TEMP_JSON, y1)
        utils.open_code(utils.TEMP_JSON)

    # parse
    ds = ds.shuffle(seed=42)
    ds.map(
        gen_parse_jsonl1, batched=False, load_from_cache_file=False, num_proc=num_proc
    )
    logger.success("gen_parse_dir done")


def gen_parse_concat_jsonl() -> None:
    # check
    fnames = sorted(sroot.PARSE_DIR.rglob("*.jsonl"))
    logger.debug(len(fnames))  # 33216

    # cmd
    cmd = f"""
    find {sroot.PARSE_DIR} -name '*.jsonl' -print0 | xargs -0 cat > {sroot.PARSE_CONCAT_JSONL}
    """.strip()
    sroot.PARSE_CONCAT_JSONL.parent.mkdir(exist_ok=True, parents=True)
    fname_sh = sroot.SCRIPT_DIR / "gen_parse_concat_jsonl.sh"
    utils.write_sh(fname_sh, cmd)

    # run
    utils.subprocess_run(f"bash {fname_sh}")
    utils.log_written(sroot.PARSE_CONCAT_JSONL)


def gen_parse_file() -> None:
    # read
    df = utils.read_df(sroot.PARSE_CONCAT_JSONL)
    df.info()
    df.sample(1).iloc[0].to_dict()

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
    gen_parse_dir()  # 1 min for 56178 files (m2 pro, 9 threads)
    gen_parse_concat_jsonl()  # 793.0M, bdb11a4b
    gen_parse_file()  # 189.5M, 6c70c29f, 370444


if __name__ == "__main__":
    if hasattr(sys, "ps1"):
        pretty.install()
        reload(utils)
        reload(sroot)
    else:
        with logger.catch(onerror=lambda _: sys.exit(1)):
            # python -m src.crawl.drri_hj.parse
            typer.run(main)
