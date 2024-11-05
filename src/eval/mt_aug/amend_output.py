import shutil
import subprocess
import sys
import time
from importlib import reload
from pathlib import Path

import typer
from datasets import Dataset
from loguru import logger
from rich import pretty
from tqdm import tqdm

import src.eval.mt_aug.dist_run.cli as dcli
import src.eval.mt_aug.dist_run.root as droot
import src.eval.mt_aug.root as sroot
from src import utils


def prepare_input(f2: Path, num_shards: int) -> None:
    # read
    df = utils.read_df(f2)
    df.sample(1).iloc[0].to_dict()

    # filter
    idx = df["pred.content"].isna()
    logger.debug(f"{idx.sum()} error")
    df = df[idx].reset_index(drop=True)
    df.sample(1).iloc[0].to_dict()

    # add output fname
    logger.debug("add output fname...")
    df["fname"] = df["id"].progress_apply(
        lambda x: str(droot.OUTPUT_DIR / f"{x}.jsonl")
    )

    # shuffle, shard, save
    ds = Dataset.from_pandas(df, preserve_index=False)
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


def trim_output_dir() -> None:
    # check
    fnames = list(droot.OUTPUT_DIR.glob("*.jsonl"))
    logger.debug(f"{len(fnames)} in {droot.OUTPUT_DIR}")

    # concat
    temp_jsonl = utils.TEMP_JSONL
    cmd = f"""
    find {droot.OUTPUT_DIR} -name '*.jsonl' -print0 | xargs -0 cat > {temp_jsonl}
    """.strip()
    temp_jsonl.parent.mkdir(exist_ok=True, parents=True)
    fname_sh = sroot.SCRIPT_DIR / "gen_temp_jsonl.sh"
    utils.write_sh(fname_sh, cmd)
    utils.subprocess_run(f"bash {fname_sh}")
    utils.log_written(temp_jsonl)

    # read
    df = utils.read_df(temp_jsonl)
    df.sample(1).iloc[0].to_dict()

    # check
    idx = df["content"].isna()
    if idx.sum() == 0:
        logger.success("no error")
        return

    # delete
    df1 = df[idx].reset_index(drop=True)
    df1["fname"] = df1["id"].apply(lambda x: str(droot.OUTPUT_DIR / f"{x}.jsonl"))
    df1.sample(1).iloc[0].to_dict()
    for fname1 in tqdm(df1["fname"], desc="delete"):
        if Path(fname1).is_file():
            _ = Path(fname1).unlink()  # type: ignore


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


def amend_output_file(f2: Path) -> None:
    # read
    df_update = utils.read_df(droot.OUTPUT_JSONL)
    df_orig = utils.read_df(f2)

    # check
    df_error = df_orig[df_orig["pred.content"].isna()].reset_index(drop=True)
    df_error.sample(1).iloc[0].to_dict()

    # update
    df_update2 = df_update.rename(columns=lambda x: f"pred.{x}" if x != "id" else x)
    df_update2.set_index("id", inplace=True)
    df = df_orig.set_index("id")
    df.update(df_update2)
    df.reset_index(inplace=True, names="id")
    df.sample(1).iloc[0].to_dict()

    # sort rows
    df.sort_values("id", inplace=True, ignore_index=True)

    # sort cols
    c1 = [c for c in df.columns if not c.startswith("meta")]
    c2 = [c for c in df.columns if c.startswith("meta")]
    cols = sorted(c1) + sorted(c2)
    df = df[cols].reset_index(drop=True)
    df.sample(1).iloc[0].to_dict()

    # check
    df_error2 = df[df["id"].isin(df_error["id"])].reset_index(drop=True)
    df_error2.sample(1).iloc[0].to_dict()
    assert df_error2["pred.content"].isna().sum() == 0, "error not fixed"

    # drop duplicate cols
    if "pred.messages" in df.columns and df["pred.messages"].equals(df["messages"]):
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
    # 결과물 빠진 것 있으면 채우기

    # config
    if 1:
        model_id = "gpt4"
        f2 = sroot.OUTPUT_GPT4_PQ  # 436.8M, acca0833, 1115091
    if 0:
        model_id = "gpt3"
        f2 = sroot.OUTPUT_GPT3_PQ

    # check
    assert f2.is_file(), f"{f2} not found"
    df = utils.read_df(f2)
    num_total = df["pred.content"].isna().sum()
    del df
    if num_total == 0:
        logger.warning("no error to fix")
        return

    check_output_dir()
    if 0:
        reset_output_dir()

    # prepare scripts
    num_shards = 36
    num_limit = -1
    gen_tmux_scripts(model_id=model_id, num_shards=num_shards, delay=3, limit=num_limit)

    # prepare data
    prepare_input(f2=f2, num_shards=num_shards)
    assert (
        len(list(droot.INPUT_DIR.glob("*.parquet"))) == num_shards
    ), "num_shards mismatch"

    # start
    stop_tmux()
    start_tmux()
    watch_output_dir(num_total=num_total, num_limit=num_limit, num_shards=num_shards)
    utils.notify(title="done", body=droot.MODEL_DIR.name)
    if 0:
        dcli
        utils.open_code(sroot.SCRIPT_DIR / "start.sh")

    # postprocess
    trim_output_dir()
    gen_output_jsonl()
    amend_output_file(f2=f2)

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
            # python -m src.eval.mt_aug.amend_output
            typer.run(main)
