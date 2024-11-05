import random
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
from datasets import Dataset
from loguru import logger
from rich import pretty
from tqdm import tqdm

import src.crawl.niu_cc_zh.root as sroot
from src import utils


def gen_url(data_id: str) -> str:
    if 0:
        data_id = "kna_108050??_005"
    # Convert the string to a URL-compatible format
    data_id2 = urllib.parse.quote(data_id)
    url = f"https://sjw.history.go.kr/id/{data_id2}"
    return url


def cmd_unzip_7z() -> None:
    sroot.CRAWL_DL_DIR.mkdir(exist_ok=True, parents=True)
    free_space = humanize.naturalsize(
        shutil.disk_usage(sroot.CRAWL_DL_DIR).free, gnu=True
    )
    cmd = f"""
    # FREE: {free_space}
    7za l {sroot.CRAWL_7Z}
    7za x {sroot.CRAWL_7Z} -o{sroot.CRAWL_DL_DIR} -y
    """
    sroot.SCRIPT_DIR.mkdir(exist_ok=True, parents=True)
    utils.write_sh(sroot.SCRIPT_DIR / "unzip_crawl_7z.sh", cmd)


def cmd_fix_unzip_dir() -> None:
    target_dir = sroot.CRAWL_DL_DIR
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
    utils.write_sh(sroot.SCRIPT_DIR / "fix_unzip_dir.sh", cmd)


def parse1(x1: dict[Any, Any]) -> list[dict[Any, Any]]:  # noqa: C901
    _fname = x1["fname"]
    if 0:
        utils.open_code(_fname)

    # read
    fname = Path(x1["fname"])
    txt1 = fname.read_text(encoding="utf-8")

    # split
    lines = txt1.strip().splitlines()
    lines = [x.strip() for x in lines if x.strip()]

    # check
    assert len(lines) % 2 == 0, f"not even: {len(lines)}"

    # parse to dataframes
    pairs_list = [(lines[i], lines[i + 1]) for i in range(0, len(lines), 2)]
    df = pd.DataFrame(pairs_list, columns=["cc", "zh"])
    df.reset_index(drop=True, inplace=True)
    df["row_idx"] = df.index + 1

    # check
    vc = df["cc"].apply(lambda x: str(x)[:2]).value_counts()
    assert len(vc) == 1, f"vc: {vc}"
    vc = df["zh"].apply(lambda x: str(x)[:2]).value_counts()
    assert len(vc) == 1, f"vc: {vc}"

    # add metadata dict to each row
    for k1, v1 in x1.items():
        df[f"x.{k1}"] = v1
    if 0:
        df.sample(1).iloc[0].to_dict()

    # return
    data_list = df.to_dict(orient="records")
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

    # get files
    fnames = list(sroot.CRAWL_DL_DIR.rglob("bitext.txt"))
    len(fnames)  # 7304
    if 0:
        utils.open_code(sroot.CRAWL_DL_DIR)
        utils.open_code(fnames[0])

    # get todo list
    ds = Dataset.from_dict({"fname": [str(p) for p in fnames]})
    x1 = ds.shuffle()[0]
    assert Path(x1["fname"]).is_file(), f"File missing: {x1['fname']}"

    # gen temp_id
    digit = len(str(len(ds)))
    ds = ds.sort("fname")
    ds = ds.map(
        function=lambda x, idx: {"temp_id": f"L{idx:0{digit}}"},
        with_indices=True,
        num_proc=num_proc,
    )
    ds.shuffle()[0]

    # gen target fname
    ds = ds.map(
        lambda x1: {"fname2": str(sroot.PARSE_DIR / f'{x1["temp_id"]}.jsonl')},
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
        # get
        x1 = ds.filter(
            lambda x: x["data_id"] == "SJW-F12030110-03000", num_proc=num_proc
        )[0]
        #
        x1 = ds[random.choice(range(100))]  # small
        x1 = ds[random.choice(range(len(ds) - 100, len(ds)))]  # large
        #
        x1 = ds.shuffle()[0]  # random
        parse1(x1=x1)
        gen_parse_jsonl1(x1=x1)
        utils.open_file(x1["fname"])
        utils.open_code(x1["fname2"])
        utils.open_url(x1["url"])
        #
        x1 = ds.shuffle()[0]  # random
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
    fnames = sorted(sroot.PARSE_DIR.rglob("*.jsonl"))
    logger.debug(len(fnames))  # 33216

    # cmd
    cmd = f"""
    find {sroot.PARSE_DIR} -name '*.jsonl' -print0 | xargs -0 cat > {sroot.PARSE_CONCAT_JSONL}
    """.strip()
    sroot.PARSE_CONCAT_JSONL.parent.mkdir(exist_ok=True, parents=True)
    fname_sh = sroot.SCRIPT_DIR / "gen_parse_concat_jsonl.sh"
    utils.write_sh(fname_sh, cmd)

    # run
    utils.subprocess_run(f"bash {fname_sh}")
    utils.log_written(sroot.PARSE_CONCAT_JSONL)


def gen_parse_file() -> None:
    # read
    df = utils.read_df(sroot.PARSE_CONCAT_JSONL)
    df.info()
    df.sample(1).iloc[0].to_dict()

    # save
    sroot.PARSE_PQ.parent.mkdir(exist_ok=True, parents=True)
    utils.write_df2(sroot.PARSE_PQ, df)
    logger.debug(f"len: {len(df)}")


def main() -> None:
    # basic parse raw html files into jsonl
    tqdm.pandas()
    cmd_unzip_7z()
    cmd_fix_unzip_dir()
    gen_parse_dir()  # 10 sec for 7304 files
    gen_parse_concat_jsonl()  # 454.2M, cdee392e
    gen_parse_file()  # 93.0M, 9407d314, 972467


if __name__ == "__main__":
    if hasattr(sys, "ps1"):
        pretty.install()
        reload(utils)
        reload(sroot)
    else:
        with logger.catch(onerror=lambda _: sys.exit(1)):
            # python -m src.crawl.niu_cc_zh.parse
            typer.run(main)