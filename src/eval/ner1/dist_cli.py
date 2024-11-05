import json
import sys
import time
from importlib import reload
from pathlib import Path
from typing import Any, Callable

import torch
import typer
from datasets import Dataset, disable_caching, load_dataset
from loguru import logger
from rich import pretty
from tqdm import tqdm

import src.eval.ner1.dist_root as droot
import src.train.ner1.ajd.model
import src.train.ner1.ajd_cc.model
import src.train.ner1.ajd_klc.model
import src.train.ner1.ajd_klc_cc.model
import src.train.ner1.cc.model
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


def run_ner(  # noqa: C901
    x: dict[str, Any],
    model_id: str,
    shard_idx: int,
    do_skip: bool = True,
) -> None:
    # load
    global model
    global device
    if 0:
        reload(src.train.ner1.cc.model)
    if model is None:
        assert isinstance(device, str)
        torch_dtype = torch.float16 if "cuda" in device else torch.float32
        if model_id == "cc":
            model = src.train.ner1.cc.model.HanjaNER(
                device=device, torch_dtype=torch_dtype
            )
        elif model_id == "ajd":
            model = src.train.ner1.ajd.model.HanjaNER(
                device=device, torch_dtype=torch_dtype
            )
        elif model_id == "ajd_cc":
            model = src.train.ner1.ajd_cc.model.HanjaNER(
                device=device, torch_dtype=torch_dtype
            )
        elif model_id == "ajd_klc":
            model = src.train.ner1.ajd_klc.model.HanjaNER(
                device=device, torch_dtype=torch_dtype
            )
        elif model_id == "ajd_klc_cc":
            model = src.train.ner1.ajd_klc_cc.model.HanjaNER(
                device=device, torch_dtype=torch_dtype
            )
        else:
            raise ValueError(f"bad model_id: {model_id}")

    assert model is not None

    # skip if all exists
    fnames = [Path(f) for f in x["fname"]]
    if do_skip and all(f.is_file() for f in fnames):
        logger.debug(f"skip: {len(fnames)}")
        return

    # inference
    pred: list[str]
    error: str | None = None
    start_time = time.time()
    try:
        pred = model.predict_batch(x=x["text"])
        assert len(pred) == len(x["text"]), "len mismatch: pred"
        assert all(isinstance(s, str) for s in pred), "type mismatch: pred"
    except Exception as e:
        logger.error(f"{repr(e)=}")
        error = repr(e)
        pred = ["error"] * len(x["text"])
    if 0:
        utils.temp_diff("\n\n".join(x["text_xml"]), "\n\n".join(pred))
    duration = (time.time() - start_time) / len(pred)

    # save
    for i, pred1 in enumerate(pred):
        fname1 = Path(x["fname"][i])
        d1 = {
            "id": x["id"][i],
            "model_id": model_id,
            "text": x["text"][i],
            "content": pred1,
            "duration": duration,
        }
        if error is not None:
            d1["error"] = error
        pred_jsonl_str = json.dumps(d1, ensure_ascii=False) + "\n"
        fname1.write_text(pred_jsonl_str, encoding="utf-8")


def main(
    shard: str = "0of1", gpu: str = "cuda:0", modelid: str = "model", limit: int = -1
) -> None:
    # for testing
    if 0:
        shard = "0of4"
        gpu = "cuda:0"
        modelid = "cc"
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
    batch_size: int = {"cc": 64}.get(model_id, 64)
    model_func: Callable = {  # type: ignore
        "cc": lambda x: run_ner(x=x, model_id=model_id, shard_idx=shard_idx),
    }.get(model_id, lambda x: run_ner(x=x, model_id=model_id, shard_idx=shard_idx))

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
        run_ner(x=x, model_id=model_id, shard_idx=shard_idx, do_skip=do_skip)
        utils.open_code(x["fname"][0])
        # test batch
        x = ds.shuffle()[:8]
        run_ner(x=x, model_id=model_id, shard_idx=shard_idx, do_skip=do_skip)
        utils.open_code(x["fname"][0])
        # read all
        df = utils.read_df(droot.INPUT_PQ)
        df["fname"] = df["id"].progress_apply(
            lambda x: str(droot.ERROR_DIR / f"{x}_{model_id}.jsonl")
        )
        # test specific
        x = df[df["id"] == "R0441323"].to_dict(orient="list")
        run_ner(x=x, model_id=model_id, shard_idx=shard_idx, do_skip=False)
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
