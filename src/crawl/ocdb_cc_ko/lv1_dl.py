import shutil
import sys
import urllib.parse
from importlib import reload
from pathlib import Path

import humanize
import pandas as pd
import psutil
import typer
from loguru import logger
from rich import pretty
from tqdm import tqdm

import src.crawl.ocdb_cc_ko.root as sroot
from src import utils


def gen_url(x1: dict) -> str:  # type: ignore
    # for pandas
    if not isinstance(x1, dict):
        x1 = x1.to_dict()

    # convert
    x2 = {k: urllib.parse.quote(str(v)) for k, v in x1.items()}
    url = f"http://mdb.cyberseodang.or.kr/mobile2/bookList/titleList.do?bnCode={x2['data_id']}"
    return url


def gen_url2(x1: dict) -> str:  # type: ignore
    # for pandas
    if not isinstance(x1, dict):
        x1 = x1.to_dict()

    # convert
    x2 = {k: urllib.parse.quote(str(v)) for k, v in x1.items()}
    url2 = f"http://db.cyberseodang.or.kr/front/alphaList/BookMain.do?bnCode={x2['data_id']}&titleId=C1"
    return url2


def gen_task_file() -> None:
    # read
    fnames = [sroot.BOOK2_JSONL]
    df_list = []
    for fname in tqdm(fnames):
        df1 = utils.read_df(fname)
        df_list.append(df1)
    df = pd.concat(df_list, ignore_index=True)
    df.drop_duplicates(subset=["data_id"], inplace=True)

    # add: url
    df["url"] = df.apply(gen_url, axis=1)
    df["url2"] = df.apply(gen_url2, axis=1)
    if 0:
        x1 = df.sample(1).iloc[0].to_dict()
        gen_url(x1=x1)
        df.sample(1).iloc[0].to_dict()
        utils.open_url(df["url"].sample(1).iloc[0])
        utils.open_url(df["url2"].sample(1).iloc[0])

    # check
    df["data_id"].apply(lambda x: len(x)).value_counts() / len(df) * 100

    # add: fname
    df.reset_index(drop=True, inplace=True)
    df["temp_id"] = df.index + 1
    digit = len(str(df["temp_id"].max()))
    df["temp_id"] = df["temp_id"].apply(lambda x: f"L{x:0{digit}}")
    df["fname"] = df["temp_id"].apply(lambda x: str(sroot.LV1_DL_DIR / f"{x}.html"))

    # sort cols
    df = df[sorted(df.columns)].reset_index(drop=True)

    # save
    sroot.LV1_TASK_PQ.parent.mkdir(exist_ok=True, parents=True)
    utils.write_df2(sroot.LV1_TASK_PQ, df)


def get_html1(x1: dict[str, str]) -> None:
    try:
        fname = Path(x1["fname"])
        url = x1["url"]
        html = utils.get_httpx(url, mobile=True).strip()
        if len(html) < 50:
            logger.warning(f"Short! | fname: {fname} | len: {len(html)} | url: {url}")
        fname.parent.mkdir(exist_ok=True, parents=True)
        fname.write_text(html, encoding="utf-8")
        return
    except Exception as e:
        logger.error(f"Exception: {repr(e)} | x1={x1}")
        return


def gen_dl_dir(pool: int) -> None:
    # set
    num_proc = psutil.cpu_count() - 1

    # read
    ds = utils.read_ds(sroot.LV1_TASK_PQ)
    logger.debug(f"{len(ds)=}")

    # skip existing
    ds = ds.filter(lambda x1: not Path(str(x1["fname"])).is_file(), num_proc=num_proc)
    logger.debug(f"{len(ds)=}")

    if 0:
        x1 = ds.shuffle()[0]
        get_html1(x1=x1)
        utils.open_file(x1["fname"])
        utils.open_url(x1["url2"])

    # download
    ds = ds.shuffle(seed=42)
    ds.map(get_html1, batched=False, load_from_cache_file=False, num_proc=pool)
    logger.success(f"DONE: {sroot.LV1_DL_DIR}")


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
        logger.debug(df.iloc[0]["fname"])
    else:
        logger.success("All files are ok")
    if 0:
        # remove
        for fname in df1["fname"]:
            logger.warning(f"Remove: {fname}")
            Path(fname).unlink(missing_ok=True)


def assert_dl_dir() -> None:
    df = utils.read_df(sroot.LV1_TASK_PQ)
    df["ok"] = df["fname"].progress_apply(lambda x: Path(x).is_file())
    assert df["ok"].all(), "Some files are missing"
    logger.success("All files are present")


def cmd_gen_7z() -> None:
    sroot.LV1_7Z.parent.mkdir(exist_ok=True, parents=True)
    free_space = humanize.naturalsize(
        shutil.disk_usage(sroot.LV1_7Z.parent).free, gnu=True
    )
    cmd = f"""
    df -h {sroot.LV1_7Z.parent} # {free_space}
    du -hd0 {sroot.LV1_DL_DIR}
    7za a {sroot.LV1_7Z} {sroot.LV1_DL_DIR}
    """
    fname = sroot.SCRIPT_DIR / "gen_7z.sh"
    fname.parent.mkdir(exist_ok=True, parents=True)
    utils.write_sh(fname, cmd)


def run_gen_7z() -> None:
    fname = sroot.SCRIPT_DIR / "gen_7z.sh"
    logger.info(f"Run: {fname}")
    utils.subprocess_run(f"bash {fname}")
    utils.log_written(sroot.LV1_7Z)


def main() -> None:
    # crawl
    if 0:
        utils.reset_dir(sroot.LV1_DL_DIR)
    try:
        gen_task_file()
        gen_dl_dir(pool=4)
        logger.debug(utils.folder_size(sroot.LV1_DL_DIR))  # 13.7M
        check_dl_dir()
        assert_dl_dir()
        cmd_gen_7z()
        run_gen_7z()  # 721.8K, 49cc1191
    finally:
        utils.notify(title="Done", body=f"{sroot.MODEL_DIR.stem}")


if __name__ == "__main__":
    tqdm.pandas()
    #
    if hasattr(sys, "ps1"):
        pretty.install()
        reload(utils)
        reload(sroot)
    else:
        with logger.catch(onerror=lambda _: sys.exit(1)):
            # tmux new -s crawl
            # conda run --no-capture-output -n mmm python -m src.crawl.ocdb_cc_ko.lv1_dl
            # tmux attach-session -t crawl
            typer.run(main)
