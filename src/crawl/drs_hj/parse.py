import random
import re
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

import src.crawl.drs_hj.root as sroot
from src import utils

IGNORE_MISSING = True


def gen_url(data_id: str) -> str:
    if 0:
        data_id = "kna_108050??_005"
    # Convert the string to a URL-compatible format
    data_id2 = urllib.parse.quote(data_id)
    url = f"https://sjw.history.go.kr/id/{data_id2}"
    return url


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


def parse1(x1: dict[Any, Any]) -> list[dict[Any, Any]]:  # noqa: C901
    _fname = x1["fname"]
    _data_id = x1["data_id"]
    _url = x1["url"]

    if 0:
        utils.open_file(_fname)
        utils.open_code(_fname)
        utils.open_url(_url)
        _data_id

    # read
    fname = Path(x1["fname"])
    html = fname.read_text(encoding="utf-8")
    soup = BeautifulSoup(html, "lxml")
    if 0:
        utils.write_str(utils.TEMP_HTML, html)
        utils.open_file(utils.TEMP_HTML)
        utils.open_code(utils.TEMP_HTML)

    # check
    if len(html) < 100:
        raise ValueError("HTML too short")

    # check
    tags = list(soup.select("p.error_tit"))
    if len(tags) >= 1:
        raise ValueError("p.error_tit in HTML")

    # check
    must_str = "idcilpboard"  # 기사ID
    if must_str not in str(soup).lower():
        raise ValueError(f"{must_str} not found (safe)")

    # data_id
    tags = list(soup.select("a"))
    tags = [t for t in tags if "idcilpboard" in str(t).lower()]
    assert len(tags) == 1, "data_id not single"
    t0 = tags[0]
    pattern = re.compile(r"SJW-[^\'\"\(\)]+")
    mat = pattern.search(t0.attrs["href"])
    assert mat is not None
    data_id = mat.group(0)

    # page_path
    len(soup.select("ul.location"))
    t = soup.select_one("ul.location")
    assert t is not None, "page_path missing"
    pat1 = re.compile(r"\s*[\n\t]+\s*")
    page_path = utils.squeeze_whites(pat1.sub(" > ", t.text.strip()))

    # page_date
    tags = list(soup.select("span.tit_loc"))
    assert len(tags) <= 1, "page_date too many"
    #
    t = soup.select_one("span.tit_loc")
    assert t is not None, "page_date missing (safe)"
    page_date = utils.squeeze_whites(t.text.strip())

    # page_title
    # if missing, it is probably a higher level page with index of items.
    assert len(soup.select("h3.ins_view_tit")) <= 1, "page_title too many"
    #
    t = soup.select_one("h3.ins_view_tit")
    assert t is not None, "page_title missing (safe)"
    page_title = utils.squeeze_whites(t.text)

    # page_btn
    tags = list(soup.select("a.btn_connect"))
    tags = [t for t in tags if t.text.strip() in {"규장각 원본", "국역"}]
    page_btn_html = {t.text.strip(): str(t.parent) for t in tags}

    # parse
    tags = list(soup.select("div.ins_view_in"))
    elems = tags
    len(elems)
    data_list: list[dict[str, str]] = []
    if 0:
        elem = elems[0]
        elem = random.choice(elems)
        utils.squeeze_whites(elem.text)
    for elem_idx, elem in enumerate(elems):
        d1: dict[str, Any] = {}
        d1["x1"] = dict(x1)
        d1["elem_idx"] = elem_idx
        d1["page_path"] = page_path
        d1["page_date"] = page_date
        d1["page_title"] = page_title
        d1["page_btn_html"] = page_btn_html

        # data_id (not for sillok)
        if 0:
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
        d1["data_id"] = data_id
        d1["url"] = gen_url(data_id)

        # title
        if 0:
            tags = list(elem.select("div.text_body_tit"))
            assert len(tags) == 1, "title not single"
            t = tags[0]
            d1["title_html"] = str(t)
            d1["title_text"] = utils.squeeze_whites(t.text)

        # dci (opt)
        if 0:
            tags = list(elem.select("div.dci-pane"))
            assert len(tags) <= 1, "dci too many"
            if len(tags) == 1:
                t = tags[0]
                d1["dci_html"] = str(t)
                d1["dci_text"] = utils.squeeze_whites(t.text)

        # body
        t = elem
        d1["body_html"] = str(t)
        d1["body_text"] = utils.squeeze_whites(t.text)

        # copyright (opt)
        if 0:
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
    ds = utils.read_ds(sroot.CRAWL_TASK_PQ)
    x1 = ds.shuffle()[0]

    # source fname: fix if needed
    if not Path(x1["fname"]).parent.is_dir():
        ds = ds.map(
            lambda x1: {"fname": str(sroot.CRAWL_DL_DIR / Path(x1["fname"]).name)},
            num_proc=num_proc,
        )

    # check
    if IGNORE_MISSING:
        ds = ds.filter(lambda x1: Path(x1["fname"]).is_file(), num_proc=num_proc)
        logger.debug(ds)
    x1 = ds.shuffle()[0]
    assert Path(x1["fname"]).is_file(), f"File missing: {x1['fname']}"

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
        x1 = ds.filter(
            lambda x: x["data_id"] == "SJW-F12030110-03000", num_proc=num_proc
        )[0]
        #
        x1 = ds[random.choice(range(100))]  # small
        x1 = ds[random.choice(range(len(ds) - 100, len(ds)))]  # large
        #
        x1 = ds.shuffle()[0]  # random
        parse1(x1=x1)
        gen_parse_jsonl1(x1=x1)
        utils.open_file(x1["fname"])
        utils.open_code(x1["fname2"])
        utils.open_url(x1["url"])
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
    logger.debug(f"len: {len(df)}")


def main() -> None:
    # basic parse raw html files into jsonl
    tqdm.pandas()
    cmd_unzip_7z()
    cmd_fix_unzip_dir()
    gen_parse_dir()  # 50 min for 1787k files (intel 39 threads)
    gen_parse_concat_jsonl()  # 5.3G
    gen_parse_file()  # 1.4G


if __name__ == "__main__":
    if hasattr(sys, "ps1"):
        pretty.install()
        reload(utils)
        reload(sroot)
    else:
        with logger.catch(onerror=lambda _: sys.exit(1)):
            # python -m src.crawl.drs_hj.parse
            typer.run(main)
