import os
import shutil
import subprocess
import sys
import time
from importlib import reload
from pathlib import Path

import typer
from datasets import Dataset, disable_caching, load_dataset
from loguru import logger
from rich import pretty
from tqdm import tqdm

import src.eval.mt_aug.dist_run.cli as dcli
import src.eval.mt_aug.dist_run.root as droot
import src.eval.mt_aug.root as sroot
from src import utils


def prepare_input(f1: Path, num_shards: int) -> None:
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
        function=lambda x: {"fname": str(droot.OUTPUT_DIR / f"{x['id']}.jsonl")},
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
    model_id: str, num_shards: int, delay: int, limit: int = -1
) -> None:
    script_list: list[str] = []
    tname_list: list[str] = []
    for i in range(num_shards):
        sleep_sec = 1 + delay * i
        shard1 = f"{i}of{num_shards}"
        script1_raw = f"python -m {dcli.__name__} --shard {shard1} --gpu cpu --modelid {model_id} --limit {limit}"
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


def watch_output_dir(num_total: int, num_limit: int, num_shards: int) -> None:
    # check
    num_prev = len(list(droot.OUTPUT_DIR.glob("*.jsonl")))
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
            num_curr = len(list(droot.OUTPUT_DIR.glob("*.jsonl")))
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
    df.sort_values("id", inplace=True, ignore_index=True)

    # sort cols
    c1 = [c for c in df.columns if not c.startswith("meta")]
    c2 = [c for c in df.columns if c.startswith("meta")]
    cols = sorted(c1) + sorted(c2)
    df = df[cols].reset_index(drop=True)
    df.sample(1).iloc[0].to_dict()

    # drop duplicate cols
    if df["pred.messages"].equals(df["messages"]):
        df.drop(columns=["pred.messages"], inplace=True)

    # save
    f2.parent.mkdir(exist_ok=True, parents=True)
    utils.write_df2(f2, df)


def prune_output_dir(f2: Path) -> None:
    # read
    df = utils.read_df(f2)
    df.sample(1).iloc[0].to_dict()

    # filter
    df1 = df[df["pred.content"].isna()].reset_index(drop=True)
    df1["fname"] = df1["id"].apply(lambda x: str(droot.OUTPUT_DIR / f"{x}.jsonl"))
    if 0:
        df1.sample(1).iloc[0].to_dict()

    # log
    num_total = len(df)
    num_error = len(df1)
    logger.debug(f"{num_error}/{num_total} error files ({num_error / num_total:.1%})")

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
    model_count = df["pred.model"].value_counts().to_dict()
    logger.debug(f"{model_count=}")

    # filter
    fallback_ratio = (df["pred.service"] == "openai").sum() / len(df)
    logger.debug(f"fallback: {fallback_ratio:.1%}")

    # time
    mean_duration = df["pred.duration"].mean()
    logger.debug(f"mean duration: {mean_duration:.1f} sec")
    total_hour_est = num_total * mean_duration / 60 / 60
    logger.debug(f"total hour: {total_hour_est:.1f} hour / 1 thread")
    logger.debug(
        f"total hour: {total_hour_est / num_shards:.1f} hour / {num_shards} thread"
    )

    # cost
    input_tokens = df["pred.prompt_tokens"].sum()
    output_tokens = df["pred.completion_tokens"].sum()
    mult = num_total / len(df)
    models = ["gpt-3.5-turbo-0125", "gpt-4-0125-preview"]
    for model in models:
        curr_usd = utils.calculate_openai_pricing(
            input_tokens=input_tokens, output_tokens=output_tokens, model=model
        )
        total_usd_est = utils.calculate_openai_pricing(
            input_tokens=input_tokens * mult,
            output_tokens=output_tokens * mult,
            model=model,
        )
        logger.debug(f"price: {curr_usd:.2f} USD | n={len(df)} | {model}")
        logger.debug(f"price: {total_usd_est:.2f} USD | n={num_total} | {model}")


def stop_tmux() -> None:
    # Stop distributed process
    start_sh = sroot.SCRIPT_DIR / "stop.sh"
    logger.debug(f"Running bash script: {start_sh}")
    subprocess.run(f"bash {start_sh}", shell=True)


def check_output_dir() -> None:
    if not droot.OUTPUT_DIR.is_dir():
        droot.OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
    num_files = len(list(droot.OUTPUT_DIR.glob("*.jsonl")))
    logger.debug(f"{num_files} files in {droot.OUTPUT_DIR}")


def reset_output_dir() -> None:
    logger.debug(f"reset {droot.OUTPUT_DIR}")
    if droot.OUTPUT_DIR.is_dir():
        shutil.rmtree(droot.OUTPUT_DIR)
    droot.OUTPUT_DIR.mkdir(exist_ok=True, parents=True)


def main() -> None:  # noqa: C901
    # input
    f1 = sroot.DATASET_PQ
    num_total = 1115091
    if 0:
        num_total = len(utils.read_df(f1))
        logger.debug(f"{num_total=}")

    # config
    if 1:
        # gpt4: 1200 sample, 12 shards, 12 min; 1200 / 12 / 12 / 60 = 0.14 RPS / thread
        # 12 threads -> 1.7 it/s
        # 36 threads, 3 key -> 5.5 it/s, 56 hours?
        # 1115091 / 5.5 / 60 / 60 = 56 hours = 2.3 days, ok.
        # 하지만 24 threads는 rate limit 걸림. 키가 더 필요함.
        # 36 threads, 3 key 로 간다.
        # 최종 결과: 60 hour, 5.5% fallback, mean 6sec, 3630USD
        model_id = "gpt4"
        f2 = sroot.OUTPUT_GPT4_PQ
    if 0:
        # gpt3: 500 sample, 5 shards, 2 min; 500 / 5 / 2 / 60 = 0.8 RPS / thread
        # 12 threads -> 7 it/s, total 43 hour, fallback 6%, 180USD (result)
        # 1800 RPM / 60 = 30 RPS -> max 30 threads. 그런데 16개도 가끔 rate limit 걸림.
        # 1115091 / 1 / 30 / 60 / 60 = 10 hours
        model_id = "gpt3"
        f2 = sroot.OUTPUT_GPT3_PQ

    # check
    if f2.is_file():
        logger.warning(f"{f2=} already exists")
    check_output_dir()
    if 0:
        reset_output_dir()

    # prepare scripts
    num_shards = 36
    num_limit = -1
    if 0:
        num_limit = 100
    gen_tmux_scripts(model_id=model_id, num_shards=num_shards, delay=3, limit=num_limit)
    if 0:
        gen_tmux_scripts(model_id=model_id, num_shards=10, delay=5, limit=1000)

    # prepare data
    if 0:
        prepare_input(f1=f1, num_shards=num_shards)
    assert (
        len(list(droot.INPUT_DIR.glob("*.parquet"))) == num_shards
    ), "num_shards mismatch"

    # start
    utils.subprocess_run("tmux ls")
    stop_tmux()
    start_tmux()
    watch_output_dir(num_total=num_total, num_limit=num_limit, num_shards=num_shards)
    if 0:
        watch_output_dir(num_total=num_total, num_limit=-1, num_shards=num_shards)
    utils.notify(title="done", body=droot.MODEL_DIR.name)
    if 0:
        dcli
        utils.open_code(sroot.SCRIPT_DIR / "start.sh")
        utils.open_code(sroot.ESTIMATION_JSON)

    # postprocess
    gen_output_jsonl()
    gen_output_file(f1=f1, f2=f2)
    check_output_file(f2=f2, num_total=num_total, num_shards=num_shards)

    # error handling
    if 0:
        prune_output_dir(f2=f2)

    # cleanup
    stop_tmux()


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
            # python -m src.eval.mt_aug.run_inference
            typer.run(main)
