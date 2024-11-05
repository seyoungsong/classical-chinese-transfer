import json
import os
import random
import shutil
import subprocess
import sys
import time
from importlib import reload
from pathlib import Path

import httpx
import pandas as pd
import typer
from datasets import Dataset, disable_caching, load_dataset
from loguru import logger
from openai import OpenAI
from rich import pretty
from tqdm import tqdm

import src.eval.mt_llm.dist_cli as dcli
import src.eval.mt_llm.dist_root as droot
import src.eval.mt_llm.root as sroot
import src.train.mt_api.vllm.model
from src import utils

MODELS_STR = """
# not needed
# anonymous/TowerInstruct-7B-v0.2-KLC-AWQ
# anonymous/TowerInstruct-7B-v0.2-AJD-KLC-CC-01to0-AWQ
# anonymous/TowerInstruct-7B-v0.2-AJD-KLC-CC-01to1-AWQ
# anonymous/TowerInstruct-7B-v0.2-AJD-KLC-CC-NoAug-AWQ

# main
Unbabel/TowerInstruct-7B-v0.2
anonymous/TowerInstruct-7B-v0.2-CC-AWQ
anonymous/TowerInstruct-7B-v0.2-AJD-AWQ
anonymous/TowerInstruct-7B-v0.2-AJD-CC-AWQ
anonymous/TowerInstruct-7B-v0.2-AJD-KLC-AWQ
anonymous/TowerInstruct-7B-v0.2-AJD-KLC-CC-AWQ

# by size
#anonymous/TowerInstruct-7B-v0.2-AJD-KLC-CC-1_32to0-AWQ
#anonymous/TowerInstruct-7B-v0.2-AJD-KLC-CC-1_32to1-AWQ

#anonymous/TowerInstruct-7B-v0.2-AJD-KLC-CC-1_16to0-AWQ
#anonymous/TowerInstruct-7B-v0.2-AJD-KLC-CC-1_16to1-AWQ

#anonymous/TowerInstruct-7B-v0.2-AJD-KLC-CC-1_8to0-AWQ
#anonymous/TowerInstruct-7B-v0.2-AJD-KLC-CC-1_8to1-AWQ

#anonymous/TowerInstruct-7B-v0.2-AJD-KLC-CC-1_4to0-AWQ
#anonymous/TowerInstruct-7B-v0.2-AJD-KLC-CC-1_4to1-AWQ

#anonymous/TowerInstruct-7B-v0.2-AJD-KLC-CC-05to0-AWQ
#anonymous/TowerInstruct-7B-v0.2-AJD-KLC-CC-05to1-AWQ

#anonymous/TowerInstruct-7B-v0.2-AJD-KLC-CC-1to0-AWQ
#anonymous/TowerInstruct-7B-v0.2-AJD-KLC-CC-1to1-AWQ

#anonymous/TowerInstruct-7B-v0.2-AJD-KLC-CC-2to0-AWQ
#anonymous/TowerInstruct-7B-v0.2-AJD-KLC-CC-2to1-AWQ
"""


def prepare_input(
    f1: Path,
    f_prev: Path,
    model_id: str,
    num_shards: int,
) -> tuple[int, int]:
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
    model_id2 = model_id.replace("/", "__")
    ds = ds.map(
        function=lambda x: {
            "fname": str(droot.OUTPUT_DIR / f"{x['id']}_{model_id2}.jsonl")
        },
        num_proc=num_proc,
        load_from_cache_file=False,
    )
    if 0:
        ds.shuffle()[0]

    # filter out if already done (exists in prev_output)
    if f_prev.is_file():
        logger.debug("filter: prev_output...")
        df_prev = utils.read_df(f_prev)
        df_prev = df_prev[df_prev["pred.model_id"] == model_id].reset_index(drop=True)
        assert df_prev["key2"].is_unique, "df_prev key2 not unique"
        logger.debug(f"{len(df_prev)} in {f_prev}")
        key2_set = set(df_prev["key2"].tolist())
        prev_len = len(ds)
        ds = ds.filter(
            function=lambda x: x["key2"] not in key2_set,
            num_proc=num_proc,
            load_from_cache_file=False,
        )
        curr_len = len(ds)
        logger.info(
            f"{prev_len - curr_len} already exist in prev file ({prev_len=}, {curr_len=}) ({model_id=})"
        )
    else:
        logger.warning(f"{f_prev} not found")

    # number of jsonl files that should exist in output_dir (num_watch)
    num_watch = len(ds)

    # filter out if already done (.jsonl exists)
    logger.debug("filter: output_dir...")
    prev_len = len(ds)
    ds = ds.filter(
        function=lambda x: not Path(x["fname"]).is_file(),
        num_proc=num_proc,
        load_from_cache_file=False,
    )
    curr_len = len(ds)
    logger.debug(f"{ds=}")
    logger.info(
        f"{prev_len - curr_len} already done in curr output dir ({prev_len=}, {curr_len=})"
    )

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

    # return
    return num_watch, curr_len


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
    num_watch: int, model_id: str, num_limit: int, num_shards: int
) -> None:
    # check
    model_id2 = model_id.replace("/", "__")
    pattern = f"*_{model_id2}.jsonl"
    num_prev = len(list(droot.OUTPUT_DIR.glob(pattern)))
    if num_limit > 0:
        num_task = min(num_watch, num_limit * num_shards)
    else:
        num_task = num_watch
    num_todo = num_task - num_prev
    logger.debug(
        f"todo: {num_todo}/{num_task} ({num_todo / num_task:.1%}) ({num_watch=})"
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


def gen_output2_file(f2: Path, f_prev: Path, f_output2: Path) -> None:
    # check
    if f_prev is None or not f_prev.is_file():
        logger.warning(f"{f_prev} not found, simply copy {f2} to {f_output2}")
        f_output2.parent.mkdir(exist_ok=True, parents=True)
        shutil.copy2(f2, f_output2)
        utils.log_written(f_output2)
        return

    # read
    df_output = utils.read_df(f2)
    df_output = df_output[df_output["pred.content"].notna()].reset_index(drop=True)

    # read
    df_prev = utils.read_df(f_prev)
    df_prev = df_prev[df_prev["pred.content"].notna()].reset_index(drop=True)

    # concat
    df = pd.concat([df_prev, df_output], ignore_index=True)
    if 0:
        df.sample(1).iloc[0].to_dict()
        df.groupby(["key2", "pred.model_id"]).size().value_counts()

    # check
    if not (df.groupby(["key2", "pred.model_id"]).size() == 1).all():
        logger.warning("drop duplicate in output2...")
        df.drop_duplicates(subset=["key2", "pred.model_id"], inplace=True)

    # sort rows
    df.sort_values(["key2", "pred.model_id"], inplace=True, ignore_index=True)

    # sort cols
    c1 = [c for c in df.columns if not c.startswith("meta")]
    c2 = [c for c in df.columns if c.startswith("meta")]
    cols = sorted(c1) + sorted(c2)
    df = df[cols].reset_index(drop=True)
    df.sample(1).iloc[0].to_dict()

    # save
    f_output2.parent.mkdir(exist_ok=True, parents=True)
    utils.write_df2(f_output2, df)


def prune_output_dir(f2: Path) -> None:
    # read
    df = utils.read_df(f2)
    df.sample(1).iloc[0].to_dict()

    # filter
    df = utils.replace_blank_to_none(df)
    df["fname"] = df.apply(
        lambda x: str(
            droot.OUTPUT_DIR
            / f'{x["id"]}_{x["pred.model_id"].replace("/", "__")}.jsonl'
        ),
        axis=1,
    )
    df1 = df[df["pred.content"].isna()].reset_index(drop=True)
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
    df["model_id"] = df["stem"].apply(lambda x: str(x).split("_", maxsplit=1)[-1])
    logger.debug(df.groupby("model_id").size())


def reset_output_dir() -> None:
    logger.debug(f"reset {droot.OUTPUT_DIR}")
    if droot.OUTPUT_DIR.is_dir():
        shutil.rmtree(droot.OUTPUT_DIR)
    droot.OUTPUT_DIR.mkdir(exist_ok=True, parents=True)


def ssh_deploy_vllm(model_id: str, stop_all: bool = False) -> None:
    auto_sh = (utils.SCRIPT_ROOT_DIR / "../guides/docker/vllm/auto.sh").resolve()
    assert auto_sh.is_file(), f"{auto_sh} not found"
    gpu_list = [d["gpu"] for d in src.train.mt_api.vllm.model.VLLM_RESOURCE_LIST]
    gpu_list_str = ",".join(map(str, gpu_list))
    if stop_all:
        gpu_list_str = "stop"
    cmd = f"""
    ssh -p 10022 anonymous@anonymous.com 'MODEL_ID="{model_id}" GPU_LIST="{gpu_list_str}" bash -s' -- < {auto_sh}
    """.strip()
    print(cmd)
    utils.subprocess_run(cmd)


def get_vllm_model(idx: int = 0) -> str:
    resource = src.train.mt_api.vllm.model.VLLM_RESOURCE_LIST[idx]
    client = OpenAI(
        api_key=resource["api_key"],
        base_url=resource["base_url"],
        http_client=httpx.Client(http2=True, verify=False, timeout=4),
    )
    models = json.loads(client.models.list().model_dump_json())
    curr_model_id = models["data"][0]["id"]
    assert isinstance(curr_model_id, str), "curr_model_id is not str"
    if 0:
        logger.debug(f"{resource['base_url']} : {curr_model_id}")
    return curr_model_id


def wait_vllm_servers(model_id: str, max_num_fail: int) -> None:
    if 0:
        get_vllm_model(idx=0)
    num_fail = 0
    while True:
        if num_fail > max_num_fail:
            utils.notify(
                title=f"{num_fail=} > {max_num_fail=}", body=droot.MODEL_DIR.name
            )
            raise RuntimeError("wait_vllm_servers: too many fail")
        try:
            vllm_list = [
                get_vllm_model(idx=i)
                for i in range(len(src.train.mt_api.vllm.model.VLLM_RESOURCE_LIST))
            ]
            assert len(set(vllm_list)) == 1, "model_id not consistent"
            assert model_id == vllm_list[0], "model_id mismatch"
            break
        except Exception as e:
            logger.warning(f"{repr(e)} ({num_fail=}/{max_num_fail=})")
            num_fail += 1
            time.sleep(1)
    logger.success(f"all servers are ready ({model_id=})")


def check_output2_file(f_output2: Path) -> None:
    # read
    df = utils.read_df(f_output2)
    df.sample(1).iloc[0].to_dict()

    # filter
    idx = df["pred.content"].isna()
    if idx.sum() > 0:
        logger.debug(df[idx].sample(1).iloc[0].to_dict()["pred.error"])
        logger.warning(
            f"drop {idx.sum()} rows with null pred.content ({idx.mean():.2%})"
        )
        df = df[~idx].reset_index(drop=True)

    # check
    df["meta.corpus"] = df["meta.corpus"] + "-" + df["lang.src"] + "-" + df["lang.tgt"]
    df1 = df.groupby(["pred.model_id", "meta.corpus"]).size()
    df1 = (
        df1.reset_index()
        .pivot_table(index="pred.model_id", columns="meta.corpus", values=0)  # type: ignore
        .fillna(0)
    )
    sroot.RESULT_DIR.mkdir(parents=True, exist_ok=True)
    fname1 = sroot.RESULT_DIR / "output2_size.tsv"
    df1.to_csv(fname1, sep="\t", index=True, encoding="utf-8-sig")
    utils.log_written(fname1)


def main(num_shards: int = 8, reset: bool = False) -> None:  # noqa: C901
    # files
    f1 = sroot.DATASET_PQ
    f2 = sroot.OUTPUT_PQ
    f_prev = sroot.PREV_OUTPUT_PQ
    f_output2 = sroot.OUTPUT2_PQ
    num_total = len(utils.read_df(f1))
    logger.debug(f"{num_total=}")  # 12000
    if f_prev.is_file():
        df_prev = utils.read_df(f_prev)
        sz_prev = df_prev.groupby(["pred.model_id"]).size()
        logger.debug(f"{sz_prev=}")
        del df_prev
        del sz_prev
    if 0:
        num_shards = 8
        #
        num_shards = 12 * len(src.train.mt_api.vllm.model.VLLM_RESOURCE_LIST)
        num_limit = 100

    # config
    model_list = [s.strip() for s in MODELS_STR.strip().split("\n") if s.strip()]
    model_list = [s.replace("-QLoRA", "-AWQ") for s in model_list if "#" not in s]
    logger.debug(f"{model_list=}")
    gpu_list = None
    num_limit = -1

    # check
    if num_shards % len(src.train.mt_api.vllm.model.VLLM_RESOURCE_LIST) != 0:
        logger.warning(
            f"{num_shards=} not multiple of len(src.train.mt_api.vllm.model.VLLM_RESOURCE_LIST)={len(src.train.mt_api.vllm.model.VLLM_RESOURCE_LIST)}"
        )

    # check
    check_output_dir()
    if reset:
        reset_output_dir()

    # for each model
    if 0:
        model_id = random.choice(model_list)
    for model_idx, model_id in tqdm(
        enumerate(model_list), desc="model", total=len(model_list)
    ):
        # debug
        logger.debug(f"{model_id=} ({model_idx+1}/{len(model_list)})")

        # prepare scripts
        gen_tmux_scripts(
            model_id=model_id,
            num_shards=num_shards,
            delay=1,
            limit=num_limit,
            gpu_list=gpu_list,
        )

        # prepare data
        num_watch, curr_len = prepare_input(
            f1=f1, f_prev=f_prev, model_id=model_id, num_shards=num_shards
        )
        if curr_len == 0:
            logger.success("no more data to process")
            continue
        assert (
            len(list(droot.INPUT_DIR.glob("*.parquet"))) == num_shards
        ), "num_shards mismatch"

        # deploy model
        ssh_deploy_vllm(model_id=model_id)

        # wait servers
        wait_vllm_servers(model_id=model_id, max_num_fail=50)

        # start
        utils.subprocess_run("tmux ls")
        stop_tmux()
        start_tmux()
        if 0:
            utils.open_code(sroot.SCRIPT_DIR / "start.sh")
            utils.open_code(droot.OUTPUT_DIR)
        watch_output_dir(
            num_watch=num_watch,
            model_id=model_id,
            num_limit=num_limit,
            num_shards=num_shards,
        )

        # cleanup
        stop_tmux()
        ssh_deploy_vllm(model_id=model_id, stop_all=True)

    # postprocess
    stop_tmux()
    ssh_deploy_vllm(model_id=model_id, stop_all=True)
    utils.notify(title="done", body=droot.MODEL_DIR.name)
    gen_output_jsonl()
    gen_output_file(f1=f1, f2=f2)
    check_output_file(f2=f2, num_total=num_total, num_shards=num_shards)

    # error handling
    prune_output_dir(f2=f2)

    # merge with prev_output
    gen_output2_file(
        f2=f2, f_prev=f_prev, f_output2=f_output2
    )  # 128.5M, ffdafd6e, 536803
    check_output2_file(f_output2=f_output2)


if __name__ == "__main__":
    tqdm.pandas()
    #
    if hasattr(sys, "ps1"):
        pretty.install()
        reload(utils)
        reload(droot)
        reload(sroot)
        reload(src.train.mt_api.vllm.model)
    else:
        with logger.catch(onerror=lambda _: sys.exit(1)):
            # python -m src.eval.mt_llm.run_inference --num-shards $((8 * 8))
            typer.run(main)
