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


def gen_url(x1: dict) -> str:  # type: ignore
    # for pandas
    if not isinstance(x1, dict):
        x1 = x1.to_dict()

    # convert
    x2 = {k: urllib.parse.quote(str(v)) for k, v in x1.items()}
    # https://kyudb.snu.ac.kr/series/subitem.do?nodeCnt=1000&book_cate=ILS&pid=4&py_king_nm=%EC%88%9C%EC%A1%B0(%EC%9D%B5%EC%A2%85)&book_cd=GK12813_00&id=1000&nlevel=2&label=%EC%88%9C%EC%A1%B0(%EC%9D%B5%EC%A2%85)29%20(1829%2C%EA%B8%B0%EC%B6%95)&py_year=1829&py_month=&py_yun=&py_day=&py_century=&vol_no=&div_menu=1&item_cd=ILS&cate=1&grp_no=&upd_yn=N
    # https://kyudb.snu.ac.kr/series/subitem.do?nodeCnt=1000&book_cate=ILS&pid=4&py_king_nm=순조(익종)&book_cd=GK12813_00&id=1000&nlevel=2&label=순조(익종)29 (1829,기축)&py_year=1829&py_month=&py_yun=&py_day=&py_century=&vol_no=&div_menu=1&item_cd=ILS&cate=1&grp_no=&upd_yn=N
    url = f"https://kyudb.snu.ac.kr/series/subitem.do?nodeCnt=1000&book_cate=ILS&pid=4&py_king_nm={x2['py_king_nm']}&book_cd={x2['book_cd']}&id=1000&nlevel=2&label={x2['title']}&py_year={x2['py_year']}&py_month=&py_yun=&py_day=&py_century=&vol_no=&div_menu=1&item_cd=ILS&cate=1&grp_no=&upd_yn=N"

    return url


def gen_task_file() -> None:
    # read
    df = utils.read_df(sroot.LV1A_PQ)
    df.sample(1).iloc[0].to_dict()

    # drop trivial
    nu = df.nunique()
    df = df[list(nu[nu > 1].index)]

    # add: url
    df["url"] = df.apply(gen_url, axis=1)
    if 0:
        x1 = df.sample(1).iloc[0].to_dict()
        gen_url(x1=x1)
        utils.open_url(df["url"].sample(1).iloc[0])

    # add: fname
    df.reset_index(drop=True, inplace=True)
    df["temp_id"] = df.index + 1
    digit = len(str(df["temp_id"].max()))
    df["temp_id"] = df["temp_id"].apply(lambda x: f"L{x:0{digit}}")
    df["fname"] = df["temp_id"].apply(lambda x: str(sroot.LV2_DL_DIR / f"{x}.html"))

    # save
    sroot.LV2_TASK_PQ.parent.mkdir(exist_ok=True, parents=True)
    utils.write_df2(sroot.LV2_TASK_PQ, df)


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
    ds = utils.read_ds(sroot.LV2_TASK_PQ)
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
    fnames = list(sroot.LV2_DL_DIR.rglob("*.html"))
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
        logger.debug(df.iloc[0]["fname"])
    else:
        logger.success("All files are ok")
    if 0:
        # remove
        for fname in df1["fname"]:
            logger.warning(f"Remove: {fname}")
            Path(fname).unlink(missing_ok=True)


def assert_dl_dir() -> None:
    df = utils.read_df(sroot.LV2_TASK_PQ)
    tqdm.pandas()
    df["ok"] = df["fname"].progress_apply(lambda x: Path(x).is_file())
    assert df["ok"].all(), "Some files are missing"
    logger.success("All files are present")


def main() -> None:
    # crawl index files
    gen_task_file()  # 16.1K, ac6a8650
    gen_dl_dir()  # 3 sec, 161
    logger.debug(utils.folder_size(sroot.LV2_DL_DIR))  # 2.4M
    check_dl_dir()  # size 0: 53
    assert_dl_dir()


if __name__ == "__main__":
    if hasattr(sys, "ps1"):
        pretty.install()
        reload(utils)
        reload(sroot)
    else:
        with logger.catch(onerror=lambda _: sys.exit(1)):
            # python -m src.crawl.drri_hj.lv2_dl
            typer.run(main)