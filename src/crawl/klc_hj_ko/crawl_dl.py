import shutil
import sys
from importlib import reload
from pathlib import Path
from typing import Any

import humanize
import pandas as pd
import typer
from loguru import logger
from rich import pretty
from tqdm import tqdm

import src.crawl.klc_hj_ko.root as sroot
from src import utils


def gen_task_file() -> None:
    # read
    df = utils.read_df(sroot.INDEX2_PQ)
    df.sample(1).iloc[0].to_dict()

    # check
    df["extra_type"].value_counts()
    df["data_id"].apply(lambda x: len(x)).value_counts() / len(df) * 100
    # 92% of data_id has length 27, rest is 22, 18

    # add: url
    df["url"] = "unk"
    df["url2"] = "unk"
    idx = df["extra_type"] == "orig"
    df.loc[idx, "url"] = df.loc[idx, "data_id"].apply(
        lambda x: f"https://db.itkc.or.kr/dir/node?dataId={x}&viewSync=OT"
    )
    df.loc[idx, "url2"] = df.loc[idx, "data_id"].apply(
        lambda x: f"https://db.itkc.or.kr/dir/item?itemId=BT#/dir/node?dataId={x}&viewSync=OT"
    )
    idx = df["extra_type"].isin(["punc", "both"])
    df.loc[idx, "url"] = df.loc[idx, "data_id"].apply(
        lambda x: f"https://db.itkc.or.kr/dir/node?dataId={x}&viewSync=KP"
    )
    df.loc[idx, "url2"] = df.loc[idx, "data_id"].apply(
        lambda x: f"https://db.itkc.or.kr/dir/item?itemId=BT#/dir/node?dataId={x}&viewSync=KP"
    )
    assert (df["url"] == "unk").sum() == 0, "url is not filled properly."

    if 0:
        utils.open_url(df["url"].sample(1).iloc[0])

    # add: fname
    df.reset_index(drop=True, inplace=True)
    df["temp_id"] = df.index + 1
    digit = len(str(df["temp_id"].max()))
    df["temp_id"] = df["temp_id"].apply(lambda x: f"L{x:0{digit}}")
    df["fname"] = df["temp_id"].apply(lambda x: str(sroot.CRAWL_DL_DIR / f"{x}.html"))

    # save
    sroot.CRAWL_TASK_PQ.parent.mkdir(exist_ok=True, parents=True)
    utils.write_df2(sroot.CRAWL_TASK_PQ, df)


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
    ds = utils.read_ds(sroot.CRAWL_TASK_PQ)
    logger.debug(f"ds={ds}")

    # remove existing
    ds = ds.filter(lambda x1: not Path(str(x1["fname"])).is_file(), num_proc=8)
    logger.debug(f"ds={ds}")

    if 0:
        # test
        x1 = ds.shuffle()[0]
        get_html1(x1=x1)
        utils.open_file(x1["fname"])
        utils.open_url(x1["url2"])

    ds = ds.shuffle(seed=42)
    ds = ds.map(get_html1, batched=False, load_from_cache_file=False, num_proc=24)
    logger.success(f"ds={ds}")


def check_dl_dir() -> None:
    # read
    fnames = list(sroot.CRAWL_DL_DIR.rglob("*.html"))
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
    logger.debug(f"len(df1)={len(df1)}")
    if 0:
        # remove
        for fname in df1["fname"]:
            logger.warning(f"Remove: {fname}")
            Path(fname).unlink(missing_ok=True)


def assert_dl_dir() -> None:
    df = utils.read_df(sroot.CRAWL_TASK_PQ)
    tqdm.pandas()
    df["ok"] = df["fname"].progress_apply(lambda x: Path(x).is_file())
    assert df["ok"].all(), "Some files are missing"
    logger.success("All files are present")


def gen_dl_7z() -> None:
    sroot.CRAWL_7Z.parent.mkdir(exist_ok=True, parents=True)
    free_space = humanize.naturalsize(
        shutil.disk_usage(sroot.CRAWL_7Z.parent).free, gnu=True
    )
    cmd = f"""
    df -h {sroot.CRAWL_7Z.parent} # {free_space}
    du -hd0 {sroot.CRAWL_DL_DIR}
    7za a {sroot.CRAWL_7Z} {sroot.CRAWL_DL_DIR}
    """
    fname = sroot.SCRIPT_DIR / "gen_crawl_7z.sh"
    fname.parent.mkdir(exist_ok=True, parents=True)
    utils.write_sh(fname, cmd)
    logger.info(f"Run: {fname}")
    utils.subprocess_run(f"bash {fname}")
    utils.log_written(sroot.CRAWL_7Z)


def main() -> None:
    if 0:
        gen_task_file()  # 6.2M, d11d0338
        gen_dl_dir()
        logger.debug(utils.folder_size(sroot.CRAWL_DL_DIR))  # 2.4G
        check_dl_dir()
        assert_dl_dir()
        gen_dl_7z()  # 300.2M, 110c36b9
    # crawl
    try:
        gen_dl_dir()  # eta 1h?
    finally:
        utils.notify(title="crawl", body=f"Done: {utils.datetime_kst().isoformat()}")


if __name__ == "__main__":
    if hasattr(sys, "ps1"):
        pretty.install()
        reload(utils)
        reload(sroot)
    else:
        with logger.catch(onerror=lambda _: sys.exit(1)):
            # python -m src.crawl.klc_hj_ko.crawl_dl
            typer.run(main)
