import sys
import urllib.parse
from importlib import reload
from pathlib import Path
from typing import Any

import pandas as pd
import typer
from loguru import logger
from rich import pretty
from tqdm import tqdm

import src.crawl.drri_hj.root as sroot
from src import utils


def gen_url(data_id: str) -> str:
    # https://kyudb.snu.ac.kr/series/subitem.do?nodeCnt=1000&book_cate=ILS&pid=0&py_king_nm=정조&book_cd=&id=2&nlevel=1&label=정조&py_year=&py_month=&py_yun=&py_day=&py_century=&vol_no=&div_menu=1&item_cd=ILS&cate=1&grp_no=&upd_yn=
    if 0:
        data_id = "순조(익종)"
    # Convert the string to a URL-compatible format
    d2 = urllib.parse.quote(data_id)
    url = f"https://kyudb.snu.ac.kr/series/subitem.do?nodeCnt=1000&book_cate=ILS&pid=0&py_king_nm={d2}&book_cd=&id=2&nlevel=1&label={d2}&py_year=&py_month=&py_yun=&py_day=&py_century=&vol_no=&div_menu=1&item_cd=ILS&cate=1&grp_no=&upd_yn="
    return url


def gen_task_file() -> None:
    # read
    df = utils.read_df(sroot.BOOK2_JSONL)
    df.sample(1).iloc[0].to_dict()

    # add: url
    df["url"] = df["data_id"].apply(gen_url)
    if 0:
        utils.open_url(df["url"].sample(1).iloc[0])

    # add: fname
    df.reset_index(drop=True, inplace=True)
    df["temp_id"] = df.index + 1
    digit = len(str(df["temp_id"].max()))
    df["temp_id"] = df["temp_id"].apply(lambda x: f"L{x:0{digit}}")
    df["fname"] = df["temp_id"].apply(lambda x: str(sroot.LV1_DL_DIR / f"{x}.html"))

    # save
    sroot.LV1_TASK_PQ.parent.mkdir(exist_ok=True, parents=True)
    utils.write_df2(sroot.LV1_TASK_PQ, df)


def get_html1(x1: dict[str, str]) -> dict[str, Any]:
    try:
        fname = Path(x1["fname"])
        url = x1["url"]
        html = utils.get_httpx(url).strip()
        fname.parent.mkdir(exist_ok=True, parents=True)
        fname.write_text(html, encoding="utf-8")
        return {"size": len(html)}
    except Exception as e:
        logger.error(f"Exception: {repr(e)} | x1={x1}")
        return {"size": -1}


def gen_dl_dir() -> None:
    # read
    ds = utils.read_ds(sroot.LV1_TASK_PQ)
    logger.debug(f"ds={ds}")

    # remove existing
    ds = ds.filter(lambda x1: not Path(str(x1["fname"])).is_file(), num_proc=5)
    logger.debug(f"ds={ds}")

    if 0:
        # test
        x1 = ds.shuffle()[0]
        get_html1(x1=x1)
        utils.open_file(x1["fname"])

    ds = ds.shuffle(seed=42)
    ds = ds.map(get_html1, batched=False, load_from_cache_file=False, num_proc=8)
    logger.success(f"ds={ds}")
    utils.notify(title="crawl", body=f"Done: {utils.datetime_kst().isoformat()}")


def check_dl_dir() -> None:
    # read
    fnames = list(sroot.LV1_DL_DIR.rglob("*.html"))
    df = pd.DataFrame({"fname": fnames})
    logger.debug(f"len(df)={len(df)}")

    # sort
    df["size"] = df["fname"].apply(lambda x: Path(x).stat().st_size)
    df.sort_values(by=["size", "fname"], inplace=True, ignore_index=True)

    # check
    logger.debug(df["size"].value_counts()[df["size"].value_counts() > 1])
    logger.debug(df["size"][:10])
    if 0:
        utils.open_file(df.iloc[0]["fname"])
        utils.open_file(df.iloc[4]["fname"])

    df1 = df[df["size"] < 100]
    if len(df1) > 0:
        logger.warning(f"len(df1)={len(df1)}")
    else:
        logger.success("All files are ok")
    if 0:
        # remove
        for fname in df1["fname"]:
            logger.warning(f"Remove: {fname}")
            Path(fname).unlink(missing_ok=True)


def assert_dl_dir() -> None:
    df = utils.read_df(sroot.LV1_TASK_PQ)
    tqdm.pandas()
    df["ok"] = df["fname"].progress_apply(lambda x: Path(x).is_file())
    assert df["ok"].all(), "Some files are missing"
    logger.success("All files are present")


def main() -> None:
    # crawl index files
    gen_task_file()  # 6.3K, 1fb4d9e5
    gen_dl_dir()  # 1 sec
    logger.debug(utils.folder_size(sroot.LV1_DL_DIR))  # 208.8K
    check_dl_dir()
    assert_dl_dir()


if __name__ == "__main__":
    if hasattr(sys, "ps1"):
        pretty.install()
        reload(utils)
        reload(sroot)
    else:
        with logger.catch(onerror=lambda _: sys.exit(1)):
            # python -m src.crawl.drri_hj.lv1_dl
            typer.run(main)
