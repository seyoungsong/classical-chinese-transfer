import shutil
import sys
from importlib import reload
from pathlib import Path
from typing import Any

import humanize
import pandas as pd
import psutil
import typer
from bs4 import BeautifulSoup
from loguru import logger
from rich import pretty
from tqdm import tqdm

import src.crawl.ocdb_cc_ko.root as sroot
from src import utils


def cmd_unzip_7z(fname: Path, unzip_dir: Path) -> None:
    unzip_dir.mkdir(exist_ok=True, parents=True)
    free_space = humanize.naturalsize(shutil.disk_usage(unzip_dir).free, gnu=True)
    cmd = f"""
    # FREE: {free_space}
    7za l {fname}
    7za x {fname} -o{unzip_dir} -y
    """
    sroot.SCRIPT_DIR.mkdir(exist_ok=True, parents=True)
    utils.write_sh(sroot.SCRIPT_DIR / "unzip_7z.sh", cmd)


def cmd_fix_unzip_dir(unzip_dir: Path) -> None:
    nested_dir = unzip_dir / unzip_dir.name
    if nested_dir.is_dir():
        logger.warning(f"nested_dir found: {nested_dir}")
    temp_dir = unzip_dir.parent / f"{unzip_dir.name}_temp"
    cmd = f"""
    mv {nested_dir} {temp_dir}
    du -hd0 {unzip_dir}
    rm -rf {unzip_dir}
    mv {temp_dir} {unzip_dir}
    """
    utils.write_sh(sroot.SCRIPT_DIR / "fix_unzip_dir.sh", cmd)


def parse1(x1: dict[Any, Any]) -> list[dict[Any, Any]]:
    if 0:
        utils.open_file(x1["fname"])
        utils.open_url(x1["url2"])

    # read
    fname1 = Path(x1["fname"])
    html = fname1.read_text(encoding="utf-8")
    soup = BeautifulSoup(html, "lxml")
    if 0:
        utils.write_str(utils.TEMP_HTML, html)
        utils.open_file(utils.TEMP_HTML)
        utils.open_code(utils.TEMP_HTML)

    # data
    d1 = {}
    d1["fname"] = x1["fname"]

    # get
    tags = soup.select("li")
    t = [t for t in tags if "data-bnname" in t.attrs][0]
    d1["bookname"] = t.attrs["data-bnname"]
    d1["data_id"] = t.attrs["data-bncode"] + "|" + t.attrs["id"]

    # get
    t = soup.select("div.cts_sbj")[0]
    d1["title.text"] = t.text.strip()
    d1["title.html"] = str(t)

    # get
    t = soup.select("div.cts_list")[0]
    d1["body.html"] = str(t)

    # convert
    data_list = [d1]

    return data_list


def gen_parse_jsonl1(x1: dict[Any, Any]) -> None:
    fname2 = Path(x1["fname2"])
    try:
        y1 = parse1(x1=x1)
    except Exception as e:
        if "safe" not in repr(e):
            logger.error(f"Exception: [ {repr(e)} ], x1: [ {dict(x1)} ]")
        y1 = [{"x1": dict(x1), "error": repr(e)}]
    fname2.parent.mkdir(parents=True, exist_ok=True)
    # jsonl for future concat
    df1 = pd.json_normalize(y1)
    df1.to_json(fname2, orient="records", lines=True, force_ascii=False)
    del df1


def gen_parse_dir(ignore_missing: bool) -> None:
    # init
    num_proc = psutil.cpu_count() - 1

    # read
    ds = utils.read_ds(sroot.LV2_TASK_PQ)

    # fname: fix if needed
    x1 = ds.shuffle()[0]
    if not Path(x1["fname"]).parent.is_dir():
        ds = ds.map(
            lambda x1: {"fname": str(sroot.LV2_DL_DIR / Path(x1["fname"]).name)},
            num_proc=num_proc,
        )

    # check
    if ignore_missing:
        ds = ds.filter(lambda x1: Path(x1["fname"]).is_file(), num_proc=num_proc)
    x1 = ds.shuffle()[0]
    assert Path(x1["fname"]).is_file(), f"File missing: {x1['fname']}"

    # gen target fname
    ds = ds.map(
        lambda x1: {
            "fname2": str(
                sroot.LV2_PARSE_DIR / Path(x1["fname"]).with_suffix(".jsonl").name
            )
        },
        num_proc=num_proc,
    )
    ds.shuffle()[0]

    # test
    if 0:
        #
        ds = ds.map(
            lambda x1: {"size": Path(x1["fname"]).stat().st_size}, num_proc=num_proc
        )
        ds = ds.sort("size")
        x1 = ds[0]
        #
        x1 = ds.shuffle()[0]  # random
        parse1(x1=x1)
        gen_parse_jsonl1(x1=x1)
        utils.open_file(x1["fname"])
        utils.open_code(x1["fname2"])
        utils.open_url(x1["url2"])

    # parse
    ds = ds.shuffle(seed=42)
    ds.map(
        gen_parse_jsonl1, batched=False, load_from_cache_file=False, num_proc=num_proc
    )
    logger.success("gen_parse_dir done")


def gen_parse_jsonl() -> None:
    # check
    fnames = sorted(sroot.LV2_PARSE_DIR.rglob("*.jsonl"))
    logger.debug(f"{len(fnames)=}")

    # cmd
    cmd = f"""
    find {sroot.LV2_PARSE_DIR} -name '*.jsonl' -print0 | xargs -0 cat > {sroot.LV2_PARSE_JSONL}
    """.strip()
    sroot.LV2_PARSE_JSONL.parent.mkdir(exist_ok=True, parents=True)
    fname_sh = sroot.SCRIPT_DIR / "gen_parse_jsonl.sh"
    utils.write_sh(fname_sh, cmd)

    # run
    utils.subprocess_run(f"bash {fname_sh}")
    utils.log_written(sroot.LV2_PARSE_JSONL)


def gen_parse_file() -> None:
    # read
    df = utils.read_df(sroot.LV2_PARSE_JSONL)
    df.info()
    df.sample(1).iloc[0].to_dict()

    # save
    sroot.LV2_PQ.parent.mkdir(exist_ok=True, parents=True)
    utils.write_df2(sroot.LV2_PQ, df)


def gen_parseA_file() -> None:
    # read
    df = utils.read_df(sroot.LV2_PQ)

    # drop duplicates
    df["data_id"].value_counts().value_counts()
    if 0:
        idx = df["data_id"].duplicated(keep=False)
        df[idx].sort_values(by=["data_id", "title"]).head(5).to_dict(orient="records")
    df.drop_duplicates(subset=["data_id"], inplace=True, ignore_index=True)

    # convert
    df["temp_id"] = df["fname"].apply(lambda x: Path(str(x)).stem)
    df.drop(columns=["fname"], inplace=True)
    df.sample(1).iloc[0].to_dict()

    # read
    df_task = utils.read_df(sroot.LV2_TASK_PQ)
    df_task["temp_id"] = df_task["fname"].apply(lambda x: Path(str(x)).stem)
    df_task.drop(columns=["fname"], inplace=True)

    # convert
    rcols = {c: f"x.{c}" for c in df_task.columns if c != "temp_id"}
    df_task.rename(columns=rcols, inplace=True)
    df_task.sample(1).iloc[0].to_dict()

    # join
    assert set(df_task.columns).intersection(df.columns) == {"temp_id"}
    df = df.merge(df_task, on="temp_id", how="left")
    df.sample(1).iloc[0].to_dict()

    # drop old col
    df.drop(columns=["temp_id"], inplace=True)

    # sort cols
    df = df[sorted(df.columns)]
    df.sample(1).iloc[0].to_dict()

    # sort rows
    assert df["data_id"].is_unique, "data_id not unique"
    df.sort_values(by=["data_id"], inplace=True, ignore_index=True)

    # save
    sroot.LV2A_PQ.parent.mkdir(exist_ok=True, parents=True)
    utils.write_df2(sroot.LV2A_PQ, df)


def main() -> None:
    # parse index files
    if 0:
        cmd_unzip_7z(fname=sroot.LV2_7Z, unzip_dir=sroot.LV2_DL_DIR)
        cmd_fix_unzip_dir(unzip_dir=sroot.LV2_DL_DIR)
        utils.reset_dir(sroot.LV2_PARSE_DIR)
    gen_parse_dir(ignore_missing=False)
    gen_parse_jsonl()
    gen_parse_file()
    gen_parseA_file()  # 61.1M, 68127e4d, 28341


if __name__ == "__main__":
    tqdm.pandas()
    #
    if hasattr(sys, "ps1"):
        pretty.install()
        reload(utils)
        reload(sroot)
    else:
        with logger.catch(onerror=lambda _: sys.exit(1)):
            # python -m src.crawl.ocdb_cc_ko.lv2_parse
            typer.run(main)
