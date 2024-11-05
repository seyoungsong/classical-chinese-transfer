import shutil
import sys
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

import src.crawl.ajd_link.root as sroot
from src import utils

IGNORE_MISSING = False


def cmd_unzip_7z() -> None:
    sroot.LV1_DL_DIR.mkdir(exist_ok=True, parents=True)
    free_space = humanize.naturalsize(
        shutil.disk_usage(sroot.LV1_DL_DIR).free, gnu=True
    )
    cmd = f"""
    # FREE: {free_space}
    7za l {sroot.LV1_7Z}
    7za x {sroot.LV1_7Z} -o{sroot.LV1_DL_DIR} -y
    """
    sroot.SCRIPT_DIR.mkdir(exist_ok=True, parents=True)
    utils.write_sh(sroot.SCRIPT_DIR / "unzip_lv4_7z.sh", cmd)


def cmd_fix_unzip_dir() -> None:
    target_dir = sroot.LV1_DL_DIR
    nested_dir = target_dir / target_dir.name
    if nested_dir.is_dir():
        logger.warning(f"nested_dir found: {nested_dir}")
    temp_dir = target_dir.parent / f"{target_dir.name}_temp"
    cmd = f"""
    mv {nested_dir} {temp_dir}
    du -hd0 {target_dir}
    rm -rf {target_dir}
    mv {temp_dir} {target_dir}
    """
    utils.write_sh(sroot.SCRIPT_DIR / "fix_lv4_unzip_dir.sh", cmd)


def parse1(x1: dict[Any, Any]) -> list[dict[Any, Any]]:
    if 0:
        _fname = x1["fname"]
        _data_id = x1["data_id"]
        _url_web = f"""https://sillok.history.go.kr/search/inspectionDayList.do?id={_data_id}"""  # fmt: off

        utils.open_file(_fname)
        utils.open_code(_fname)
        utils.open_url(_url_web)

    # read
    fname = Path(x1["fname"])
    html = fname.read_text(encoding="utf-8")
    if len(html) == 0:
        html = "Empty!"
    d1 = {"text": html, **x1}

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


def gen_parse_dir() -> None:
    # init
    num_proc = psutil.cpu_count() - 1

    # get todo list
    ds = utils.read_ds(sroot.LV1_TASK_PQ)
    x1 = ds.shuffle()[0]

    # source fname: fix if needed
    if not Path(x1["fname"]).parent.is_dir():
        ds = ds.map(
            lambda x1: {"fname": str(sroot.LV1_DL_DIR / Path(x1["fname"]).name)},
            num_proc=num_proc,
        )

    # check
    if IGNORE_MISSING:
        ds = ds.filter(lambda x1: Path(x1["fname"]).is_file(), num_proc=num_proc)
    x1 = ds.shuffle()[0]
    assert Path(x1["fname"]).is_file(), f"File missing: {x1['fname']}"

    # gen target fname
    ds = ds.map(
        lambda x1: {
            "fname2": str(
                sroot.LV1_PARSE_DIR / Path(x1["fname"]).with_suffix(".jsonl").name
            )
        },
        num_proc=num_proc,
    )
    ds.shuffle()[0]

    # sort by size
    ds = ds.map(
        lambda x1: {"size": Path(x1["fname"]).stat().st_size}, num_proc=num_proc
    )
    ds = ds.sort("size")

    # test
    if 0:
        x1 = ds[0]
        x1 = ds.shuffle()[0]  # random
        parse1(x1=x1)
        gen_parse_jsonl1(x1=x1)
        utils.open_file(x1["fname"])
        utils.open_code(x1["fname2"])
        utils.open_url(x1["url"])
        #
        y1 = pd.json_normalize(parse1(x1)).to_dict(orient="records")
        utils.write_json(utils.TEMP_JSON, y1)
        utils.open_code(utils.TEMP_JSON)

    # parse
    ds = ds.shuffle(seed=42)
    ds.map(
        gen_parse_jsonl1, batched=False, load_from_cache_file=False, num_proc=num_proc
    )
    logger.success("gen_parse_dir done")


def gen_parse_concat_jsonl() -> None:
    # check
    fnames = sorted(sroot.LV1_PARSE_DIR.rglob("*.jsonl"))
    logger.debug(len(fnames))

    # cmd
    cmd = f"""
    find {sroot.LV1_PARSE_DIR} -name '*.jsonl' -print0 | xargs -0 cat > {sroot.LV1_PARSE_CONCAT_JSONL}
    """.strip()
    sroot.LV1_PARSE_CONCAT_JSONL.parent.mkdir(exist_ok=True, parents=True)
    fname_sh = sroot.SCRIPT_DIR / "gen_lv1_parse_concat_jsonl.sh"
    utils.write_sh(fname_sh, cmd)

    # run
    utils.subprocess_run(f"bash {fname_sh}")
    utils.log_written(sroot.LV1_PARSE_CONCAT_JSONL)


def gen_parse_file() -> None:
    # read
    df = utils.read_df(sroot.LV1_PARSE_CONCAT_JSONL)
    df.info()
    df.sample(1).iloc[0].to_dict()

    # save
    sroot.LV1_PQ.parent.mkdir(exist_ok=True, parents=True)
    utils.write_df2(sroot.LV1_PQ, df)
    logger.debug(f"len: {len(df)}")


def gen_parseA_file() -> None:
    # read
    df = utils.read_df(sroot.LV1_PQ)

    # drop duplicates
    if 0:
        df["data_id"].value_counts().value_counts()
        df.drop_duplicates(subset=["data_id"], inplace=True, ignore_index=True)

    # drop columns
    df.drop(columns=["fname", "fname2", "size", "temp_id"], inplace=True)
    df.sample(1).iloc[0].to_dict()

    # sort cols
    df = df[sorted(df.columns)]
    df.sample(1).iloc[0].to_dict()

    # sort rows
    kcols = ["data_id"]
    df.sort_values(kcols, inplace=True, ignore_index=True)

    # save
    sroot.LV1A_PQ.parent.mkdir(exist_ok=True, parents=True)
    utils.write_df2(sroot.LV1A_PQ, df)
    logger.debug(f"len: {len(df)}")


def main() -> None:
    # parse index files
    if 0:
        cmd_unzip_7z()
        cmd_fix_unzip_dir()
        utils.reset_dir(sroot.LV1_PARSE_DIR)
    gen_parse_dir()
    gen_parse_concat_jsonl()
    gen_parse_file()
    gen_parseA_file()  # 389.8K, de3255f5, 51376


if __name__ == "__main__":
    tqdm.pandas()
    #
    if hasattr(sys, "ps1"):
        pretty.install()
        reload(utils)
        reload(sroot)
    else:
        with logger.catch(onerror=lambda _: sys.exit(1)):
            # python -m src.crawl.ajd_link.lv1_parse
            typer.run(main)
