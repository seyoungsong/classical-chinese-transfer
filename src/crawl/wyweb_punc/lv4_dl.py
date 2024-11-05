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

import src.crawl.wyweb_punc.root as sroot
from src import utils


def gen_url(data_id: str) -> str:
    # opt1: https://sjw.history.go.kr/search/inspectionDayList.do?treeID=SJW-F06030280
    # opt2: https://sjw.history.go.kr/m/kinglistday.do?kid=SJW-F06030280
    if 0:
        data_id = "SJW-F06030280"
    # Convert the string to a URL-compatible format
    data_id2 = urllib.parse.quote(data_id)
    url = f"https://sjw.history.go.kr/m/kinglistday.do?kid={data_id2}"
    return url


def gen_task_file() -> None:
    tqdm.pandas()

    # read
    df = utils.read_df(sroot.LV1A_PQ)
    df.sample(1).iloc[0].to_dict()

    # add: url
    df["url"] = df["data_id"].apply(gen_url)

    # check
    if 0:
        x1 = df.sample(1).iloc[0].to_dict()
        x1
        utils.open_url(x1["url"])

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
    # init
    num_proc = psutil.cpu_count() - 1

    # read
    ds = utils.read_ds(sroot.LV4_TASK_PQ)
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


def check_dl_dir(rm_small: bool = False, cut_size: int = 8500) -> None:
    tqdm.pandas()

    # read
    fnames = list(sroot.LV4_DL_DIR.rglob("*.html"))
    df = pd.DataFrame({"fname": fnames})
    logger.debug(f"len(df)={len(df)}")
    df["fname"] = df["fname"].apply(str)
    df["temp_id"] = df["fname"].apply(lambda x: Path(x).stem)

    # join
    df_task = utils.read_df(sroot.LV4_TASK_PQ)
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
        df1 = df[df["size"] <= cut_size]
        df1["size"].value_counts()
        logger.warning(f"del files smaller than {cut_size}: {len(df1)}")
        for fname1 in tqdm(df1["fname"]):
            Path(fname1).unlink(missing_ok=True)

    # sample
    idx = df["size"].isin(vc.index[:5])
    df1 = df[idx].groupby("size").apply(lambda x: x.sample(1)).reset_index(drop=True)
    df1.sort_values(by=["size", "fname"], inplace=True, ignore_index=True)
    d1 = df1[["size", "fname", "url"]].to_dict(orient="records")
    logger.debug(d1)


def assert_dl_dir() -> None:
    df = utils.read_df(sroot.LV4_TASK_PQ)
    tqdm.pandas()
    df["ok"] = df["fname"].progress_apply(lambda x: Path(x).is_file())
    if not df["ok"].all():
        logger.warning("Some files are missing")
        df1 = df[~df["ok"]]
        logger.warning(f"Missing: {len(df1)} ({len(df1)/len(df):.1%})")
    else:
        logger.success("All files are present")


def cmd_gen_7z() -> None:
    sroot.LV4_7Z.parent.mkdir(exist_ok=True, parents=True)
    free_space = humanize.naturalsize(
        shutil.disk_usage(sroot.LV4_7Z.parent).free, gnu=True
    )
    cmd = f"""
    df -h {sroot.LV4_7Z.parent} # {free_space}
    du -hd0 {sroot.LV4_DL_DIR}
    7za a {sroot.LV4_7Z} {sroot.LV4_DL_DIR}
    """
    fname = sroot.SCRIPT_DIR / "gen_LV4_7Z.sh"
    fname.parent.mkdir(exist_ok=True, parents=True)
    utils.write_sh(fname, cmd)


def run_gen_7z() -> None:
    fname = sroot.SCRIPT_DIR / "gen_LV4_7Z.sh"
    logger.info(f"Run: {fname}")
    utils.subprocess_run(f"bash {fname}")


def main() -> None:
    # crawl all index pages
    if 0:
        gen_task_file()  # 1.3M, 9b3e6504, 104608
        gen_dl_dir()  # eta 2h
        logger.debug(utils.folder_size(sroot.LV4_DL_DIR))  # 5.7G
        check_dl_dir()
        if 0:
            check_dl_dir(rm_small=True, cut_size=8500)
        assert_dl_dir()
        cmd_gen_7z()
        run_gen_7z()
        utils.log_written(sroot.LV4_7Z)  # 285.4M, 8eda96d6
    try:
        # tmux new -s crawl
        # conda run --no-capture-output -n mmm python -m src.crawl.wyweb_punc.lv4_dl
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
            # python -m src.crawl.wyweb_punc.lv4_dl
            typer.run(main)
