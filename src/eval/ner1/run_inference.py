import os
import shutil
import subprocess
import sys
import time
from importlib import reload
from pathlib import Path

import pandas as pd
import typer
from datasets import Dataset, disable_caching, load_dataset
from loguru import logger
from rich import pretty
from tqdm import tqdm

import src.eval.ner1.dist_cli as dcli
import src.eval.ner1.dist_root as droot
import src.eval.ner1.root as sroot
from src import utils


def prepare_input(f1: Path, model_id: str, num_shards: int) -> None:
    # config
    num_proc: int = round(os.cpu_count() * 0.9)  # type: ignore

    # original input for debugging
    logger.debug(f"f1={str(f1)}")
    assert f1.suffix == droot.INPUT_PQ.suffix, "suffix mismatch"
    droot.INPUT_PQ.parent.mkdir(exist_ok=True, parents=True)
    shutil.copy2(f1, droot.INPUT_PQ)
    utils.log_written(droot.INPUT_PQ)

    # read
    disable_caching()
    utils.log_reading(f1)
    ds: Dataset = load_dataset(
        "parquet",
        data_files={"simple": str(f1)},
        split="simple",
        download_mode="force_redownload",
        verification_mode="no_checks",
    )
    logger.debug(f"{ds=}")

    # add output fname
    logger.debug("add output fname...")
    ds = ds.map(
        function=lambda x: {
            "fname": str(droot.OUTPUT_DIR / f"{x['id']}_{model_id}.jsonl")
        },
        num_proc=num_proc,
        load_from_cache_file=False,
    )
    if 0:
        ds.shuffle()[0]

    # filter out if already done
    logger.debug("filter...")
    prev_len = len(ds)
    ds = ds.filter(
        function=lambda x: not Path(x["fname"]).is_file(),
        num_proc=num_proc,
        load_from_cache_file=False,
    )
    curr_len = len(ds)
    logger.debug(f"{ds=}")
    logger.debug(f"{prev_len - curr_len} already done ({prev_len=}, {curr_len=})")

    # shuffle, shard, save
    ds = ds.shuffle(seed=42)
    utils.reset_dir(droot.INPUT_DIR)
    if 0:
        shard_idx = 0
    for shard_idx in tqdm(range(num_shards), desc="shard"):
        fname1 = droot.INPUT_DIR / f"{shard_idx}.parquet"
        ds1 = ds.shard(num_shards=num_shards, index=shard_idx)
        ds1.to_parquet(fname1, compression="zstd")
        utils.log_written(fname1, etc=str(len(ds1)))
        del ds1

    # check
    assert (
        len(list(droot.INPUT_DIR.glob("*.parquet"))) == num_shards
    ), "num_shards mismatch"
    logger.success(f"prepared {num_shards} shards in {droot.INPUT_DIR}")
    del ds


def gen_tmux_scripts(
    model_id: str,
    num_shards: int,
    delay: int,
    limit: int = -1,
    gpu_list: list[int] | None = None,
) -> None:
    # check
    gpu_map: dict[int, str] | None = None
    if gpu_list is not None:
        assert len(gpu_list) == num_shards, "gpu_list mismatch"
        gpu_map = {i: f"cuda:{gpu_list[i]}" for i in range(num_shards)}
    gpu_map

    script_list: list[str] = []
    tname_list: list[str] = []
    for i in range(num_shards):
        sleep_sec = 1 + delay * i
        shard1 = f"{i}of{num_shards}"
        gpu1 = gpu_map[i] if gpu_map is not None else "cpu"
        script1_raw = f"python -m {dcli.__name__} --shard {shard1} --gpu {gpu1} --modelid {model_id} --limit {limit}"
        tname1 = f"shard_{shard1}_{droot.MODEL_DIR.name}"
        tname_list.append(tname1)
        script1 = f"""
        tmux new-session -d -s {tname1}
        tmux send-keys -t {tname1} "sleep {sleep_sec} && cd {Path.cwd().resolve()} && conda run --no-capture-output -n mmm {script1_raw}; if [ \\$? -eq 0 ]; then tmux kill-session -t {tname1}; fi" C-m
        # tmux attach-session -t {tname1}
        # tmux kill-session -t {tname1}
        """.strip()
        script1 += "\n\n"
        script_list.append(script1)

    # start.sh
    start_sh = sroot.SCRIPT_DIR / "start.sh"
    start_sh.parent.mkdir(parents=True, exist_ok=True)
    script_start = f"# bash {start_sh}\n" + f"# ls -hal {droot.OUTPUT_DIR}\n\n"
    script_start += "\n\n".join(script_list)
    utils.write_sh(start_sh, script_start)

    # stop.sh
    stop_sh = sroot.SCRIPT_DIR / "stop.sh"
    script_stop = f"# bash {stop_sh}\n\n"
    script_stop += "\n\n".join([f"tmux kill-session -t {s}" for s in tname_list])
    utils.write_sh(stop_sh, script_stop)


def start_tmux() -> None:
    # Start distributed process
    start_sh = sroot.SCRIPT_DIR / "start.sh"
    logger.debug(f"Running bash script: {start_sh}")
    subprocess.run(f"bash {start_sh}", shell=True)


def watch_output_dir(
    num_total: int, model_id: str, num_limit: int, num_shards: int
) -> None:
    # check
    pattern = f"*_{model_id}.jsonl"
    num_prev = len(list(droot.OUTPUT_DIR.glob(pattern)))
    if num_limit > 0:
        num_task = min(num_total, num_limit * num_shards)
    else:
        num_task = num_total
    num_todo = num_task - num_prev
    logger.debug(
        f"todo: {num_todo}/{num_task} ({num_todo / num_task:.1%}) ({num_total=})"
    )

    # watch
    with tqdm(total=num_todo) as pbar:
        while True:
            # check
            num_curr = len(list(droot.OUTPUT_DIR.glob(pattern)))
            num_done = num_curr - num_prev
            if num_done == num_todo:
                logger.success(f"done: {num_done}/{num_todo}")
                break
            # update
            pbar.n = num_done
            _ = pbar.refresh()  # type: ignore
            # sleep
            time.sleep(1)


def gen_output_jsonl() -> None:
    # check
    fnames = list(droot.OUTPUT_DIR.glob("*.jsonl"))
    logger.debug(f"{len(fnames)} in {droot.OUTPUT_DIR}")

    # cmd
    cmd = f"""
    find {droot.OUTPUT_DIR} -name '*.jsonl' -print0 | xargs -0 cat > {droot.OUTPUT_JSONL}
    """.strip()
    droot.OUTPUT_JSONL.parent.mkdir(exist_ok=True, parents=True)
    fname_sh = sroot.SCRIPT_DIR / "gen_output_jsonl.sh"
    utils.write_sh(fname_sh, cmd)

    # run
    utils.subprocess_run(f"bash {fname_sh}")
    utils.log_written(droot.OUTPUT_JSONL)


def gen_output_file(f1: Path, f2: Path) -> None:
    # read
    df_output = utils.read_df(droot.OUTPUT_JSONL)
    df_input = utils.read_df(f1)

    # left join
    df = df_output.rename(columns=lambda x: f"pred.{x}" if x != "id" else x)
    df = df.merge(df_input, on="id", how="left")
    df.sample(1).iloc[0].to_dict()

    # sort rows
    df.sort_values(["id", "pred.model_id"], inplace=True, ignore_index=True)

    # sort cols
    c1 = [c for c in df.columns if not c.startswith("meta")]
    c2 = [c for c in df.columns if c.startswith("meta")]
    cols = sorted(c1) + sorted(c2)
    df = df[cols].reset_index(drop=True)
    df.sample(1).iloc[0].to_dict()

    # drop cols
    cols = [c for c in df.columns if not c.startswith("pred.")]
    for col1 in cols:
        if f"pred.{col1}" in df.columns and df[f"pred.{col1}"].equals(df[col1]):
            df.drop(columns=[f"pred.{col1}"], inplace=True)

    # save
    f2.parent.mkdir(exist_ok=True, parents=True)
    utils.write_df2(f2, df)


def prune_output_dir(f2: Path) -> None:
    # read
    df = utils.read_df(f2)
    df.sample(1).iloc[0].to_dict()

    # filter
    df = utils.replace_blank_to_none(df)
    df1 = df[df["pred.content"].isna()].reset_index(drop=True)
    df1["fname"] = df1.apply(
        lambda x: str(droot.OUTPUT_DIR / f"{x['id']}_{x['pred.model_id']}.jsonl"),
        axis=1,
    )
    if 0:
        df1.sample(1).iloc[0].to_dict()

    # log
    num_total = len(df)
    num_error = len(df1)
    logger.debug(f"{num_error}/{num_total} error files ({num_error / num_total:.1%})")
    if num_error == 0:
        logger.success("no error files")
        return

    # move to error dir
    droot.ERROR_DIR.mkdir(exist_ok=True, parents=True)
    for fname1_str in tqdm(df1["fname"], desc="move"):
        fname1 = Path(fname1_str)
        fname2 = droot.ERROR_DIR / fname1.name
        if fname1.is_file():
            _ = shutil.move(str(fname1), str(fname2))


def check_output_file(f2: Path, num_total: int, num_shards: int) -> None:
    # read
    df = utils.read_df(f2)
    df.sample(1).iloc[0].to_dict()

    # model
    model_id_count = df["pred.model_id"].value_counts().to_dict()
    logger.debug(f"{model_id_count=}")

    # time
    mean_duration = df["pred.duration"].mean()
    logger.debug(f"mean_duration: {mean_duration:.4f} sec")
    total_min_est = num_total * mean_duration / 60
    logger.debug(f"total_min_est: {total_min_est:.1f} min / 1 thread")
    logger.debug(
        f"total_min_est: {total_min_est / num_shards:.1f} min / {num_shards} thread"
    )


def stop_tmux() -> None:
    # Stop distributed process
    start_sh = sroot.SCRIPT_DIR / "stop.sh"
    logger.debug(f"Running bash script: {start_sh}")
    subprocess.run(f"bash {start_sh}", shell=True)


def check_output_dir() -> None:
    if not droot.OUTPUT_DIR.is_dir():
        droot.OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
    # total
    fnames = list(droot.OUTPUT_DIR.glob("*.jsonl"))
    num_files = len(fnames)
    logger.debug(f"{num_files} files in {droot.OUTPUT_DIR}")
    # per model
    stems = [f.stem for f in fnames]
    df = pd.DataFrame(stems, columns=["stem"])
    df["model_id"] = df["stem"].apply(lambda x: str(x).split("_")[-1])
    model_id_count = df.groupby("model_id").size().to_dict()
    logger.debug(f"{model_id_count=}")


def reset_output_dir() -> None:
    logger.debug(f"reset {droot.OUTPUT_DIR}")
    if droot.OUTPUT_DIR.is_dir():
        shutil.rmtree(droot.OUTPUT_DIR)
    droot.OUTPUT_DIR.mkdir(exist_ok=True, parents=True)


def main() -> None:  # noqa: C901
    # files
    f1 = sroot.DATASET_PQ
    f2 = sroot.OUTPUT_PQ
    num_total = 6000
    if 0:
        num_total = len(utils.read_df(f1))
        logger.debug(f"{num_total=}")

    # config
    model_list = [
        "cc",
        "ajd",
        "ajd_cc",
        "ajd_klc",
        "ajd_klc_cc",
    ]
    gpu_list = [0, 1, 2, 3, 4, 5, 6]
    num_shards = len(gpu_list)
    num_limit = -1
    if 0:
        num_limit = 100

    # check
    check_output_dir()
    if 0:
        reset_output_dir()

    # for each model
    for model_id in tqdm(model_list, desc="model"):
        # debug
        if 0:
            model_id = "cc"

        # prepare scripts
        gen_tmux_scripts(
            model_id=model_id,
            num_shards=num_shards,
            delay=2,
            limit=num_limit,
            gpu_list=gpu_list,
        )

        # prepare data
        prepare_input(f1=f1, model_id=model_id, num_shards=num_shards)
        assert (
            len(list(droot.INPUT_DIR.glob("*.parquet"))) == num_shards
        ), "num_shards mismatch"

        # start
        utils.subprocess_run("tmux ls")
        stop_tmux()
        start_tmux()
        if 0:
            utils.open_code(sroot.SCRIPT_DIR / "start.sh")
        watch_output_dir(
            num_total=num_total,
            model_id=model_id,
            num_limit=num_limit,
            num_shards=num_shards,
        )

        # cleanup
        stop_tmux()

    # postprocess
    stop_tmux()
    utils.notify(title="done", body=droot.MODEL_DIR.name)
    gen_output_jsonl()
    gen_output_file(f1=f1, f2=f2)
    check_output_file(f2=f2, num_total=num_total, num_shards=num_shards)

    # error handling
    prune_output_dir(f2=f2)


if __name__ == "__main__":
    tqdm.pandas()
    #
    if hasattr(sys, "ps1"):
        pretty.install()
        reload(utils)
        reload(droot)
        reload(sroot)
    else:
        with logger.catch(onerror=lambda _: sys.exit(1)):
            # python -m src.eval.ner1.run_inference
            typer.run(main)
