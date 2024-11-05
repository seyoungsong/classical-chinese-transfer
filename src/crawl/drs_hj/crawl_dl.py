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
from loguru import logger
from rich import pretty
from tqdm import tqdm

import src.crawl.drs_hj.root as sroot
from src import utils

VALID_LEN = 9990
VALID_SIZE = 9990


def gen_url(data_id: str) -> str:
    if 0:
        data_id = "SJW-F04050140-00400"
    # Convert the string to a URL-compatible format
    data_id2 = urllib.parse.quote(data_id)
    url = f"https://sjw.history.go.kr/id/{data_id2}"
    return url


def gen_task_file() -> None:
    tqdm.pandas()

    # read
    fnames = [sroot.LV4A_PQ]
    df_list = []
    for fname in tqdm(fnames):
        df1 = utils.read_df(fname)
        df_list.append(df1)
    df = pd.concat(df_list, ignore_index=True)
    df.drop_duplicates(subset=["data_id"], inplace=True)

    # add: url
    df["url"] = df["data_id"].apply(gen_url)

    # check
    if 0:
        df["data_id"].apply(lambda x: x[:5]).value_counts()
        idx = df["data_id"].apply(lambda x: str(x).startswith("w"))
        idx = df["data_id"].apply(lambda x: "?" in x or "%" in x)
        idx.sum()
        df[idx].sample(1).iloc[0].to_dict()
        df[idx]["data_id"]

    # keep only leaf nodes
    df.info()
    df.sample(1).iloc[0].to_dict()
    df["parent_id"] = df["data_id"].apply(lambda x: str(x).rsplit("-", maxsplit=1)[0])
    idx = df["data_id"].isin(df["parent_id"])
    logger.debug(f"not leaf: {idx.mean():.2%} ({idx.sum()})")
    df = df[~idx].reset_index(drop=True)
    logger.debug(len(df))
    df.drop(columns=["parent_id"], inplace=True)

    # check
    df["data_id"].apply(lambda x: len(x)).value_counts() / len(df) * 100
    df["temp_len"] = df["data_id"].apply(lambda x: len(x))
    df.groupby("temp_len").sample(1)["url"].to_list()
    if 0:
        idx = df["data_id"].apply(len) < 18
        df1 = df[idx].reset_index(drop=True)
        df1.sample(1).iloc[0].to_dict()
        # https://sjw.history.go.kr/id/SJW-H06120200-
        df1["page_title"].value_counts()

    # drop minorities
    idx = df["data_id"].apply(len) < 18
    logger.debug(f"len(df)={len(df)} | minority: {idx.mean():.2%} ({idx.sum()})")
    df = df[~idx].reset_index(drop=True)
    df["data_id"].apply(lambda x: len(x)).value_counts() / len(df) * 100

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
    logger.debug(f"len: {len(df)}")

    if 0:
        utils.reset_dir(sroot.CRAWL_DL_DIR)


def get_html1(x1: dict[str, str]) -> dict[str, Any]:
    if 0:
        url = "https://sjw.history.go.kr/id/SJW-F04050140-10400"
    try:
        fname = Path(x1["fname"])
        url = x1["url"]
        # SJW: error is 9241
        html = utils.get_httpx(url, min_len=VALID_LEN, max_retry=10).strip()
        if len(html) < 100:
            logger.warning(f"Short! | fname: {fname} | len: {len(html)} | url: {url}")
        fname.parent.mkdir(exist_ok=True, parents=True)
        fname.write_text(html, encoding="utf-8")
        return {"size": len(html)}
    except Exception as e:
        logger.error(f"Exception: {repr(e)} | x1={x1}")
        return {"size": -1}


def gen_dl_dir() -> None:
    # init
    num_proc = psutil.cpu_count() - 1

    # read
    ds = utils.read_ds(sroot.CRAWL_TASK_PQ)
    logger.debug(f"ds={ds}")

    # check
    x1 = ds.shuffle()[0]
    workdir = str(Path(x1["fname"]).resolve().parent)
    logger.debug(f"workdir: {workdir}")

    # skip existing
    ds = ds.filter(lambda x1: not Path(str(x1["fname"])).is_file(), num_proc=num_proc)
    logger.debug(f"ds={ds}")

    if 0:
        # test
        x1 = ds.shuffle()[0]
        get_html1(x1=x1)
        utils.open_file(x1["fname"])
        utils.open_url(x1["url"])

    ds = ds.shuffle(seed=42)
    ds = ds.map(get_html1, batched=False, load_from_cache_file=False, num_proc=num_proc)
    logger.success(f"ds={ds}")


def check_dl_dir(rm_small: bool = False, min_size: int = VALID_SIZE) -> None:
    tqdm.pandas()

    # read
    fnames = list(sroot.CRAWL_DL_DIR.rglob("*.html"))
    df = pd.DataFrame({"fname": fnames})
    logger.debug(f"len(df)={len(df)}")
    df["fname"] = df["fname"].apply(str)
    df["temp_id"] = df["fname"].apply(lambda x: Path(x).stem)

    # join
    df_task = utils.read_df(sroot.CRAWL_TASK_PQ)
    df_task.drop(columns=["fname"], inplace=True)
    df = df.merge(df_task, on="temp_id", how="left")
    df.sample(1).iloc[0].to_dict()

    # add: size
    df["size"] = df["fname"].progress_apply(lambda x: Path(x).stat().st_size)

    # check
    df1 = df[df["size"] < 100]
    if len(df1) > 0:
        logger.warning(f"too small: {len(df1)}")
        logger.debug(df1.iloc[0].to_dict())

    # check
    vc = df["size"].value_counts()
    logger.debug(vc[vc > 1])
    if vc[:5].sum() > 500:
        logger.warning(f"{vc[:5]}")
    if rm_small:
        # remove small files (possibly error pages)
        df1 = df[df["size"] <= min_size]
        df1["size"].value_counts()
        logger.warning(
            f"del files smaller than {min_size}: {len(df1)} ({len(df1) / len(df):.1%})"
        )
        for fname1 in tqdm(df1["fname"]):
            Path(fname1).unlink(missing_ok=True)

    # sample
    idx = df["size"].isin(vc.index[:5])
    df1 = df[idx].groupby("size").apply(lambda x: x.sample(1)).reset_index(drop=True)
    df1.sort_values(by=["size", "fname"], inplace=True, ignore_index=True)
    d1 = df1[["size", "fname", "url"]].to_dict(orient="records")
    logger.debug(d1)


def assert_dl_dir() -> None:
    df = utils.read_df(sroot.CRAWL_TASK_PQ)
    tqdm.pandas()
    df["ok"] = df["fname"].progress_apply(lambda x: Path(x).is_file())
    if not df["ok"].all():
        logger.warning("Some files are missing")
        df1 = df[~df["ok"]]
        logger.warning(f"Missing: {len(df1)} ({len(df1)/len(df):.1%})")
    else:
        logger.success("All files are present")


def cmd_gen_7z() -> None:
    sroot.CRAWL_7Z.parent.mkdir(exist_ok=True, parents=True)
    free_space = humanize.naturalsize(
        shutil.disk_usage(sroot.CRAWL_7Z.parent).free, gnu=True
    )
    cmd = f"""
    df -h {sroot.CRAWL_7Z.parent} # {free_space}
    du -hd0 {sroot.CRAWL_DL_DIR}
    7za a {sroot.CRAWL_7Z} {sroot.CRAWL_DL_DIR}
    """
    fname = sroot.SCRIPT_DIR / "gen_CRAWL_7Z.sh"
    fname.parent.mkdir(exist_ok=True, parents=True)
    utils.write_sh(fname, cmd)


def run_gen_7z() -> None:
    fname = sroot.SCRIPT_DIR / "gen_CRAWL_7Z.sh"
    logger.info(f"Run: {fname}")
    utils.subprocess_run(f"bash {fname}")
    utils.log_written(sroot.CRAWL_7Z)


def main() -> None:
    # crawl all data pages
    if 0:
        gen_task_file()  # 420.8M, 800ae4fe, 1787007
        gen_dl_dir()  # eta 4h
        logger.debug(utils.folder_size(sroot.CRAWL_DL_DIR))  # 58.4G
        check_dl_dir()
        if 0:
            check_dl_dir(rm_small=True, min_size=VALID_SIZE)
        assert_dl_dir()
        cmd_gen_7z()
        run_gen_7z()
        utils.log_written(sroot.CRAWL_7Z)  # 518.0M, 2a2d4763
    try:
        # tmux new -s crawl
        # conda run --no-capture-output -n mmm python -m src.crawl.drs_hj.crawl_dl
        # tmux attach-session -t crawl
        gen_dl_dir()
    finally:
        utils.notify(title="crawl", body=f"Done: {utils.datetime_kst().isoformat()}")


if __name__ == "__main__":
    if hasattr(sys, "ps1"):
        pretty.install()
        reload(utils)
        reload(sroot)
    else:
        with logger.catch(onerror=lambda _: sys.exit(1)):
            # python -m src.crawl.drs_hj.crawl_dl
            typer.run(main)
