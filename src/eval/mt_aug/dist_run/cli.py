import json
import sys
from importlib import reload
from pathlib import Path
from typing import Any, Callable

import torch
import typer
from datasets import Dataset, disable_caching, load_dataset
from loguru import logger
from rich import pretty
from tqdm import tqdm

import src.eval.mt_aug.dist_run.root as droot
import src.train.mt_api.gpt4.model
import src.train.mt_api.gpt4.model_azure
from src import utils

try:
    model_id  # type: ignore
except NameError:
    model_id: str = ""


try:
    model  # type: ignore
except NameError:
    model: Any = None


try:
    device  # type: ignore
except NameError:
    device: Any = None


def run_gpt(
    x: dict[str, Any],
    model_id: str,
    shard_idx: int,
    do_skip: bool = True,
) -> None:
    # load
    global model
    global device
    if 0:
        reload(src.train.mt_api.gpt4.model)
    if model is None:
        location_list = src.train.mt_api.gpt4.model_azure.get_locations(
            model_id=model_id
        )
        location = location_list[shard_idx % len(location_list)]
        model = src.train.mt_api.gpt4.model.HanjaTranslator(location=location)
    assert isinstance(model, src.train.mt_api.gpt4.model.HanjaTranslator)

    # check batch size
    assert len(x["messages"]) == 1, "batch size must be 1"

    # skip if exists
    fname1 = Path(x["fname"][0])
    if do_skip and fname1.is_file():
        logger.debug(f"skip: {fname1}")
        return

    # inference
    pred: dict[str, Any]
    try:
        pred = model.chat_str(
            model_id=model_id,
            msg1_str=x["messages"][0],
            temperature=0.7,
            timeout=60,
            stream=False,
        )
    except Exception as e:
        logger.error(f"{repr(e)=}")
        pred = {"error": repr(e)}
    if 0:
        utils.temp_diff(x["text.tgt"][0], pred["content"])
        model.chat_str(
            model_id=model_id,
            msg1_str=x["messages"][0],
            temperature=0.7,
            timeout=20,
            stream=True,
        )

    # save
    fname1 = Path(x["fname"][0])
    fname1.parent.mkdir(parents=True, exist_ok=True)
    pred["id"] = x["id"][0]
    pred["messages"] = x["messages"][0]
    pred_jsonl_str = json.dumps(pred, ensure_ascii=False) + "\n"
    fname1.write_text(pred_jsonl_str, encoding="utf-8")


def main(
    shard: str = "0of1", gpu: str = "cuda:0", modelid: str = "model", limit: int = -1
) -> None:
    # for testing
    if 0:
        shard = "0of5"
        gpu = "cpu"
        modelid = "gpt3"
        limit = 100

    # shard
    assert "of" in shard, f"bad shard: {shard}"
    shard_idx = int(shard.split("of")[0])
    num_shards = int(shard.split("of")[1])
    assert 0 <= shard_idx < num_shards, f"bad shard: {shard}"

    # fname
    fname_input = droot.INPUT_DIR / f"{shard_idx}.parquet"
    assert fname_input.is_file(), f"not found: {fname_input}"

    # device
    global device
    if "cuda" in gpu and not torch.cuda.is_available():
        logger.warning("cuda not available; using cpu")
        device = "cpu"
    else:
        device = gpu

    # model
    global model_id
    model_id = modelid
    batch_size: int = {"gpt3": 1, "gpt4": 1}[model_id]
    model_func: Callable = {  # type: ignore
        "gpt3": lambda x: run_gpt(
            x=x, model_id="gpt-3.5-turbo-0125", shard_idx=shard_idx
        ),
        "gpt4": lambda x: run_gpt(
            x=x, model_id="gpt-4-0125-preview", shard_idx=shard_idx
        ),
    }[model_id]

    # log
    logger.debug(f"{shard_idx=}, {num_shards=}, {device=}")
    logger.debug(f"{model_id=}, {model_func.__name__=}")
    logger.debug(f"{limit=}, {batch_size=}")

    # dataset
    disable_caching()
    utils.log_reading(fname_input)
    ds: Dataset = load_dataset(
        "parquet",
        data_files={"simple": str(fname_input)},
        split="simple",
        download_mode="force_redownload",
        verification_mode="no_checks",
    )
    logger.debug(f"{ds=}")

    # limit
    if limit > 0:
        logger.debug(f"select {limit}...")
        ds = ds.select(range(limit))
        logger.debug(f"{ds=}")

    # debug
    if 0:
        # test 1
        x = ds.shuffle()[:1]
        do_skip = False
        run_gpt(
            x=x, model_id="gpt-3.5-turbo-0125", shard_idx=shard_idx, do_skip=do_skip
        )
        utils.open_code(x["fname"][0])
        # test batch
        x = ds.shuffle()[:8]
        run_gpt(
            x=x, model_id="gpt-4-0125-preview", shard_idx=shard_idx, do_skip=do_skip
        )
        utils.open_code(x["fname"][0])
        # read all
        df = utils.read_df(droot.INPUT_PQ)
        df["fname"] = df["id"].progress_apply(
            lambda x: str(droot.DEBUG_DIR / f"{x}.jsonl")
        )
        # test specific
        x = df[df["id"] == "R0441323"].to_dict(orient="list")
        run_gpt(x=x, model_id="gpt-3.5-turbo-0125", shard_idx=shard_idx, do_skip=False)
        utils.open_code(x["fname"][0])

    # inference
    utils.set_seed()
    ds = ds.map(
        function=model_func,
        batched=True,
        batch_size=batch_size,
        load_from_cache_file=False,
    )

    # finish
    logger.success("Done!")
    if 0:
        utils.open_code(droot.OUTPUT_DIR)


if __name__ == "__main__":
    tqdm.pandas()
    #
    if hasattr(sys, "ps1"):
        pretty.install()
        reload(utils)
    else:
        with logger.catch(onerror=lambda _: sys.exit(1)):
            typer.run(main)
