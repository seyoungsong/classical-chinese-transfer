import shutil
import sys
import urllib.parse
from importlib import reload
from pathlib import Path
from typing import Any

import humanize
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
    # https://kyudb.snu.ac.kr/series/rightItemView.do?nCnt=1000&pid=1000&py_king_nm=%EC%A0%95%EC%A1%B0&book_cd=GK12811_00&id=1000&nlevel=4&py_year=1780&py_month=06&py_yun=0&py_day=16&py_century=&vol_no=0088&div_menu=1&item_cd=ILS&cate=1&upd_yn=&searchString=&kwd=
    # https://kyudb.snu.ac.kr/series/rightItemView.do?nCnt=1000&pid=1000&py_king_nm=정조&book_cd=GK12811_00&id=1000&nlevel=4&py_year=1780&py_month=06&py_yun=0&py_day=16&py_century=&vol_no=0088&div_menu=1&item_cd=ILS&cate=1&upd_yn=&searchString=&kwd=
    url = f"https://kyudb.snu.ac.kr/series/rightItemView.do?nCnt=1000&pid=1000&py_king_nm={x2['py_king_nm']}&book_cd={x2['book_cd']}&id=1000&nlevel=4&py_year={x2['py_year']}&py_month={x2['py_month']}&py_yun={x2['py_yun']}&py_day={x2['py_day']}&py_century=&vol_no={x2['vol_no']}&div_menu=1&item_cd=ILS&cate=1&upd_yn=&searchString=&kwd="

    return url


def gen_task_file() -> None:
    # read
    df = utils.read_df(sroot.LV3A_PQ)
    df.sample(1).iloc[0].to_dict()

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
    df["fname"] = df["temp_id"].apply(lambda x: str(sroot.LV4_DL_DIR / f"{x}.html"))

    # save
    sroot.LV4_TASK_PQ.parent.mkdir(exist_ok=True, parents=True)
    utils.write_df2(sroot.LV4_TASK_PQ, df)
    logger.debug(f"len: {len(df)}")

    if 0:
        utils.reset_dir(sroot.LV4_DL_DIR)


def get_html1(x1: dict[str, str]) -> dict[str, Any]:
    try:
        fname = Path(x1["fname"])
        url = x1["url"]
        html = utils.get_httpx(url).strip()
        if len(html) < 100:
            logger.warning(f"Short! | fname: {fname} | len: {len(html)} | url: {url}")
        fname.parent.mkdir(exist_ok=True, parents=True)
        fname.write_text(html, encoding="utf-8")
        return {"size": len(html)}
    except Exception as e:
        logger.error(f"Exception: {repr(e)} | x1={x1}")
        return {"size": -1}


def gen_dl_dir() -> None:
    # read
    ds = utils.read_ds(sroot.LV4_TASK_PQ)
    logger.debug(f"ds={ds}")

    # skip existing
    ds = ds.filter(lambda x1: not Path(str(x1["fname"])).is_file(), num_proc=5)
    logger.debug(f"ds={ds}")

    if 0:
        # test
        x1 = ds.shuffle()[0]
        get_html1(x1=x1)
        utils.open_file(x1["fname"])
        utils.open_url(x1["url"])

    ds = ds.shuffle(seed=42)
    ds = ds.map(get_html1, batched=False, load_from_cache_file=False, num_proc=16)
    logger.success(f"ds={ds}")
    utils.notify(title="crawl", body=f"Done: {utils.datetime_kst().isoformat()}")


def check_dl_dir() -> None:
    # read
    fnames = list(sroot.LV4_DL_DIR.rglob("*.html"))
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
    df = utils.read_df(sroot.LV4_TASK_PQ)
    tqdm.pandas()
    df["ok"] = df["fname"].progress_apply(lambda x: Path(x).is_file())
    assert df["ok"].all(), "Some files are missing"
    logger.success("All files are present")


def gen_dl_7z() -> None:
    sroot.LV4_7Z.parent.mkdir(exist_ok=True, parents=True)
    free_space = humanize.naturalsize(
        shutil.disk_usage(sroot.LV4_7Z.parent).free, gnu=True
    )
    cmd = f"""
    df -h {sroot.LV4_7Z.parent} # {free_space}
    du -hd0 {sroot.LV4_DL_DIR}
    7za a {sroot.LV4_7Z} {sroot.LV4_DL_DIR}
    """
    fname = sroot.SCRIPT_DIR / "gen_lv4_7z.sh"
    fname.parent.mkdir(exist_ok=True, parents=True)
    utils.write_sh(fname, cmd)
    logger.info(f"Run: {fname}")
    utils.subprocess_run(f"bash {fname}")
    utils.log_written(sroot.LV4_7Z)


def main() -> None:
    # crawl index files
    gen_task_file()  # 410.4K, 4e23a015, 56178
    gen_dl_dir()  # 16 min

    logger.debug(utils.folder_size(sroot.LV4_DL_DIR))  # 2.5G
    check_dl_dir()
    assert_dl_dir()
    gen_dl_7z()  # 85.4M, 1d4a1d8f


if __name__ == "__main__":
    if hasattr(sys, "ps1"):
        pretty.install()
        reload(utils)
        reload(sroot)
    else:
        with logger.catch(onerror=lambda _: sys.exit(1)):
            # python -m src.crawl.drri_hj.lv4_dl
            typer.run(main)