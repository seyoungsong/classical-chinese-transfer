import shutil
import sys
from importlib import reload
from pathlib import Path
from typing import Any

import humanize
import pandas as pd
import typer
from bs4 import BeautifulSoup
from loguru import logger
from rich import pretty
from tqdm import tqdm

import src.crawl.mqs_hj.root as sroot
from src import utils


def cmd_gen_mqs_hj_html_7z() -> None:
    free_space = humanize.naturalsize(
        shutil.disk_usage(sroot.MQS_HJ_HTML_7Z.parent).free, gnu=True
    )
    cmd = f"""
    # FREE: {free_space}
    du -hd0 {sroot.TEMP_MQS_HJ_HTML_DIR}
    7za a {sroot.MQS_HJ_HTML_7Z} {sroot.TEMP_MQS_HJ_HTML_DIR}
    """
    sroot.SCRIPT_DIR.mkdir(exist_ok=True, parents=True)
    utils.write_sh(sroot.SCRIPT_DIR / "gen_mqs_hj_html_7z.sh", cmd)


def cmd_unzip_mqs_hj_html_7z() -> None:
    sroot.TEMP_MQS_HJ_HTML_DIR.mkdir(exist_ok=True, parents=True)
    free_space = humanize.naturalsize(
        shutil.disk_usage(sroot.TEMP_MQS_HJ_HTML_DIR).free, gnu=True
    )
    cmd = f"""
    # FREE: {free_space}
    7za l {sroot.MQS_HJ_HTML_7Z}
    7za x {sroot.MQS_HJ_HTML_7Z} -o{sroot.TEMP_MQS_HJ_HTML_DIR} -y
    """
    sroot.SCRIPT_DIR.mkdir(exist_ok=True, parents=True)
    utils.write_sh(sroot.SCRIPT_DIR / "unzip_mqs_hj_html_7z.sh", cmd)


def parse_html(html: str) -> dict[str, Any]:
    soup = BeautifulSoup(html, "lxml")

    # check error page
    if len(html) < 100:
        raise ValueError(f"HTML too short: {len(html)}")
    not_found_str = "해당하는 데이터가 없습니다"
    if not_found_str in soup.text.strip():
        raise ValueError(not_found_str)

    if 0:
        utils.write_str(utils.TEMP_HTML, str(soup))
        utils.open_file(utils.TEMP_HTML)
        utils.open_code(utils.TEMP_HTML)

    # data
    data1: dict[str, Any] = {}

    # data_id, url
    tags = list(soup.select("a"))
    tags = [t for t in tags if "URL" in t.text]
    if len(tags) == 0:
        raise ValueError("URL not found in html")
    assert len(tags) == 1
    data_id = tags[0].attrs["href"].split("'")[1]
    data1["data_id"] = data_id
    data1["url"] = f"https://sillok.history.go.kr/mc/id/{data_id}"

    # date_hj
    t = soup.select_one("span.tit_loc")
    assert t is not None
    data1["date_hj"] = utils.squeeze_whites(t.text.strip())
    t.decompose()

    # body_hj
    tags = soup.select("div.ins_view_in p.paragraph")
    data1["body_hj"] = "\n\n".join([t.text.strip() for t in tags]).strip()

    # body_hj_html
    tags = soup.select("div.ins_view_in")
    assert len(tags) == 1
    t = tags[0]
    data1["body_hj_html"] = str(t)

    return data1


def parse_fname(fname: str) -> dict[str, Any]:
    try:
        html = Path(fname).read_text()
        data1 = parse_html(html)
        return data1
    except Exception as e:
        data_id = Path(fname).stem
        url = f"https://sillok.history.go.kr/mc/id/{data_id}"
        logger.error(f"fname: {fname} | Exception: {repr(e)} | url: {url}")
        data1 = {"data_id": data_id, "error": repr(e), "url": url}
        return data1


def test_parser() -> None:
    # read
    df0 = pd.DataFrame()
    df0["fname"] = [
        str(p.resolve()) for p in sroot.TEMP_MQS_HJ_HTML_DIR.rglob("*.html")
    ]
    df0["size"] = df0["fname"].apply(lambda x: Path(x).stat().st_size)
    df0.sort_values("size", inplace=True, ignore_index=True)
    logger.debug(f"n={len(df0)}")

    # fname: error?
    fname = df0.iloc[1]["fname"]
    # fname: small
    fname = df0[df0["size"] > df0.iloc[1]["size"]].iloc[0]["fname"]
    # fname: large
    fname = df0.iloc[-1]["fname"]
    # fname: random
    fname = df0.sample(n=1)["fname"].iloc[0]
    parse_fname(fname)

    # parse one
    html = Path(fname).read_text()
    utils.write_str(utils.TEMP_HTML, html)
    utils.open_file(utils.TEMP_HTML)
    data1 = parse_html(html)
    utils.open_url(data1["url"])

    # parse many
    fnames: list[str] = df0.sample(n=50)["fname"].to_list()
    fnames += df0.head(n=50)["fname"].to_list()
    fnames += df0.tail(n=50)["fname"].to_list()
    results: list[dict[str, Any]] = []
    for fname in tqdm(fnames):
        data1 = parse_fname(fname)
        results.append(data1)
    df = pd.DataFrame(results)
    logger.debug(f"len(df)={len(df)}")

    # sort
    df.sort_values("data_id", inplace=True, ignore_index=True)
    cols: list[str] = sorted(list(df.columns))
    df = df[cols]

    # sample
    df.drop(columns=[c for c in df.columns if "html" in str(c)], inplace=True)
    for col in [s for s in cols if "url" in s]:
        idx = df[col].notnull()
        df.loc[idx, col] = df.loc[idx, col].apply(lambda x: f" {str(x).strip()} ")
    utils.sample_df(df)


def get_fnames_cache(reset: bool = False) -> list[str]:
    utils.TEMP_DIR.mkdir(exist_ok=True, parents=True)
    temp_pkl = utils.TEMP_DIR / "mqs_hj_fnames.pkl.zst"

    if reset:
        temp_pkl.unlink(missing_ok=True)

    if temp_pkl.is_file():
        df = utils.read_df(temp_pkl)
        df.sort_values("fname", inplace=True, ignore_index=True)
    else:
        df = pd.DataFrame()
        df["fname"] = [
            str(p.resolve()) for p in sroot.TEMP_MQS_HJ_HTML_DIR.rglob("*.html")
        ]
        df.sort_values("fname", inplace=True, ignore_index=True)
        utils.write_df(temp_pkl, df)

    fnames = df["fname"].to_list()
    logger.debug(f"fnames={len(fnames)}")
    return fnames


def gen_mqs_hj_src_pkl() -> None:
    # get todo list
    fnames = get_fnames_cache(reset=False)
    fnames = utils.shuffle_list(fnames, seed=42)

    # parse
    results = utils.pool_map(func=parse_fname, xs=fnames)
    df = pd.DataFrame(results)
    logger.debug(f"len(df)={len(df)}")

    # sort
    df.sort_values("data_id", inplace=True, ignore_index=True)
    cols: list[str] = sorted(list(df.columns))
    df = df[cols]

    # save
    utils.write_df(sroot.MQS_HJ_SRC_PKL, df)
    utils.log_written(sroot.MQS_HJ_SRC_PKL)

    if 0:
        df = utils.read_df(sroot.MQS_HJ_SRC_PKL)
        utils.sample_df(df)


def main() -> None:
    if 0:
        cmd_gen_mqs_hj_html_7z()
        cmd_unzip_mqs_hj_html_7z()
        test_parser()
        gen_mqs_hj_src_pkl()  # 6 min / 40 core
    gen_mqs_hj_src_pkl()


if __name__ == "__main__":
    if hasattr(sys, "ps1"):
        pretty.install()
        reload(utils)
    else:
        with logger.catch(onerror=lambda _: sys.exit(1)):
            # python -m src.crawl.mqs_hj_parse
            typer.run(main)
