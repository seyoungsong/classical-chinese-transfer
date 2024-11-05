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

import src.crawl.drri_ko.root as sroot
from src import utils


def gen_url2(data_id: str) -> str:
    item_id = data_id.split("_")[1]
    url2 = f"https://db.itkc.or.kr/dir/item?itemId={item_id}#/dir/node?dataId={data_id}"
    return url2


def gen_task_file() -> None:
    tqdm.pandas()

    # read
    fnames = [sroot.LV1A_PQ]
    df_list = []
    for fname in tqdm(fnames):
        df1 = utils.read_df(fname)
        df_list.append(df1)
    df = pd.concat(df_list, ignore_index=True)
    df.drop_duplicates(subset=["data_id"], inplace=True)

    # keep only leaf nodes
    df.info()
    df.sample(1).iloc[0].to_dict()
    df["parent_id"] = df["data_id"].apply(lambda x: str(x).rsplit("_", maxsplit=1)[0])
    idx = df["data_id"].isin(df["parent_id"])
    logger.debug(f"not leaf: {idx.mean():.2%}")  # 5.86%
    df = df[~idx].reset_index(drop=True)
    logger.debug(len(df))  # 552965
    df.drop(columns=["parent_id"], inplace=True)

    # check
    # 99.8% of data_id has length 28, rest is 22
    df["data_id"].apply(lambda x: len(x)).value_counts() / len(df) * 100
    if 0:
        idx = df["data_id"].apply(len) < 28
        df1 = df[idx].reset_index(drop=True)
        df1["url2"] = df1["data_id"].apply(gen_url2)
        df1.sample(1).iloc[0].to_dict()
        df1["page_title"].value_counts()  # 요목, 일기청 관원

    # drop minorities
    idx = df["data_id"].apply(len) < 28
    df = df[~idx].reset_index(drop=True)
    df["data_id"].apply(lambda x: len(x)).value_counts() / len(df) * 100

    # for royal, we can use parent_id for batch download
    df["parent_id"] = df["data_id"].apply(lambda x: str(x).rsplit("_", maxsplit=1)[0])
    df["parent_id"].nunique()  # 33216
    df.drop_duplicates(subset=["parent_id"], inplace=True, ignore_index=True)
    df["data_id"] = df["parent_id"]
    df.drop(columns=["parent_id"], inplace=True)
    df.sample(1).iloc[0].to_dict()
    df["data_id"].apply(lambda x: len(x)).value_counts() / len(df) * 100

    # add: url
    df["url"] = df["data_id"].apply(
        lambda x: f"https://db.itkc.or.kr/dir/node?dataId={x}"
    )
    df["url2"] = df["data_id"].progress_apply(gen_url2)

    if 0:
        utils.open_url(df["url"].sample(1).iloc[0])
        utils.open_url(df["url2"].sample(1).iloc[0])

    # add: fname
    df.reset_index(drop=True, inplace=True)
    df["temp_id"] = df.index + 1
    digit = len(str(df["temp_id"].max()))
    df["temp_id"] = df["temp_id"].apply(lambda x: f"L{x:0{digit}}")
    df["fname"] = df["temp_id"].apply(lambda x: str(sroot.CRAWL_DL_DIR / f"{x}.html"))

    # save
    sroot.CRAWL_TASK_PQ.parent.mkdir(exist_ok=True, parents=True)
    utils.write_df2(sroot.CRAWL_TASK_PQ, df)
    logger.debug(f"len(df)={len(df)}")


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
    # crawl all data pages
    if 0:
        gen_task_file()  # 552.5K, 65a4b8ff, 15385
        gen_dl_dir()
        logger.debug(utils.folder_size(sroot.CRAWL_DL_DIR))  # 330.8M
        check_dl_dir()
        assert_dl_dir()
        gen_dl_7z()  # 34.0M, 5630a470
    try:
        # tmux new -s crawl_dl
        # conda run --no-capture-output -n mmm python -m src.crawl.drri_ko.crawl_dl
        gen_dl_dir()  # eta 3min
    finally:
        utils.notify(title="crawl", body=f"Done: {utils.datetime_kst().isoformat()}")


if __name__ == "__main__":
    if hasattr(sys, "ps1"):
        pretty.install()
        reload(utils)
        reload(sroot)
    else:
        with logger.catch(onerror=lambda _: sys.exit(1)):
            # python -m src.crawl.drri_ko.crawl_dl
            typer.run(main)
