import sys
from importlib import reload
from pathlib import Path
from typing import Any

import pandas as pd
import typer
from loguru import logger
from rich import pretty
from tqdm import tqdm

import src.crawl.ajd_en.root as sroot
import src.crawl.ajd_hj_oko.root
from src import utils


def gen_task_file() -> None:
    # read
    df = utils.read_df(sroot.LV3A_PQ)
    df.sample(1).iloc[0].to_dict()
    df["data_id"].str[:4].value_counts()

    # read
    df_hj = utils.read_df(src.crawl.ajd_hj_oko.root.FORMAT2_PQ)
    df_hj.sample(1).iloc[0].to_dict()
    # range
    df_hj["meta.data_id"].str[:4].value_counts()
    df_hj = df_hj[df_hj["meta.data_id"].str.startswith("kda")].reset_index(drop=True)
    # hj
    df_hj["lang"].value_counts()
    df_hj = df_hj[df_hj["lang"] == "hj"].reset_index(drop=True)
    # get
    df_extra = pd.DataFrame()
    df_extra["data_id"] = df_hj["meta.data_id"].apply(lambda x: "e" + x[1:]).values
    df_extra.drop_duplicates(inplace=True, ignore_index=True)
    # merge
    df = pd.concat([df, df_extra], ignore_index=True)
    df.drop_duplicates(subset="data_id", inplace=True, ignore_index=True)
    df.sort_values(by="data_id", inplace=True, ignore_index=True)

    # add: url
    df["url"] = df["data_id"].apply(
        lambda x: f"http://esillok.history.go.kr/record/getDetailViewAjax.do?id={x}&sillokViewType=EngKor&lang=ko"
    )
    if 0:
        x1 = df.sample(1).iloc[0].to_dict()
        x1 = df[df["book_id"].isna()].sample(1).iloc[0].to_dict()
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
    if 0:
        url = "http://esillok.history.go.kr/record/getDetailViewAjax.do?id=eda_10911024_104&sillokViewType=EngKor&lang=ko"
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

    # remove existing
    ds = ds.filter(lambda x1: not Path(str(x1["fname"])).is_file(), num_proc=5)
    logger.debug(f"ds={ds}")

    if 0:
        # test
        x1 = ds.shuffle()[0]
        get_html1(x1=x1)
        utils.open_file(x1["fname"])

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


def main() -> None:
    # crawl index files
    gen_task_file()  # 347.0K, cfa6656e, 31309
    gen_dl_dir()  # 6 min

    logger.debug(utils.folder_size(sroot.LV4_DL_DIR))  # 206.5M
    check_dl_dir()
    assert_dl_dir()


if __name__ == "__main__":
    if hasattr(sys, "ps1"):
        pretty.install()
        reload(utils)
        reload(sroot)
    else:
        with logger.catch(onerror=lambda _: sys.exit(1)):
            # python -m src.crawl.ajd_en.lv4_dl
            typer.run(main)