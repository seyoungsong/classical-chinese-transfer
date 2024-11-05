import json
import sys
from importlib import reload
from typing import Any

import typer
from loguru import logger
from pandarallel import pandarallel
from rich import pretty
from tqdm import tqdm

import src.eval.mt_aug.root as sroot
from src import utils

TIKTOKEN_MODEL_ID = "gpt-4-0613"
ZH2KO_RATIO = 1.42


def calc_time_min_by_tpm(output_tokens: int, model: str) -> float:
    # Azure API
    output_tpm: float
    if model == "gpt-4-0125-preview":
        output_tpm = 80_000
    elif model == "gpt-3.5-turbo-0125":
        output_tpm = 300_000
    else:
        raise ValueError(f"model={model} not found")
    total_min = output_tokens / output_tpm
    return round(total_min, 1)


def calc_time_min_by_rpm(num_samples: int, model: str) -> float:
    # Azure API
    output_rpm: float
    if model == "gpt-4-0125-preview":
        output_rpm = 480
    elif model == "gpt-3.5-turbo-0125":
        output_rpm = 1800
    else:
        raise ValueError(f"model={model} not found")
    total_min = num_samples / output_rpm
    return round(total_min, 1)


def gen_estimation_json() -> None:
    # read
    df0 = utils.read_df(sroot.DATASET_PQ)
    df = df0.copy()
    df.sample(1).iloc[0].to_dict()

    # count
    df["msg_tokens"] = df["messages"].progress_apply(
        lambda x: utils.num_tiktoken_from_messages(
            messages=json.loads(x), model=TIKTOKEN_MODEL_ID
        )
    )

    # mt_aug는 text.tgt이 더 유의미하므로
    df["src_tokens"] = df["text.tgt"].progress_apply(
        lambda x: utils.num_tiktoken(s=x, model=TIKTOKEN_MODEL_ID)
    )

    # save
    d1: dict[str, Any] = {}
    d1["num_sample"] = len(df["msg_tokens"])
    d1["msg_tokens_stat"] = df["msg_tokens"].describe().round(1).to_dict()
    d1["msg_tokens"] = df["msg_tokens"].sum()
    d1["src_tokens"] = df["src_tokens"].sum()
    if 0:
        d1 = utils.read_json(sroot.ESTIMATION_JSON)

    # estimation
    d1["tgt_tokens_est"] = round(d1["src_tokens"] * ZH2KO_RATIO)

    # cost
    models = ["gpt-4-0125-preview", "gpt-3.5-turbo-0125"]
    d1["cost_usd"] = {
        model: utils.calculate_openai_pricing(
            input_tokens=d1["src_tokens"],
            output_tokens=d1["tgt_tokens_est"],
            model=model,
        )
        for model in models
    }

    # time
    d1["time_hour_by_tpm"] = {
        model: round(
            calc_time_min_by_tpm(output_tokens=d1["tgt_tokens_est"], model=model) / 60,
            1,
        )
        for model in models
    }
    d1["time_hour_by_rpm"] = {
        model: round(
            calc_time_min_by_rpm(num_samples=d1["num_sample"], model=model) / 60, 1
        )
        for model in models
    }

    # save
    sroot.ESTIMATION_JSON.parent.mkdir(parents=True, exist_ok=True)
    utils.write_json(sroot.ESTIMATION_JSON, d1)


def check_mt_ratio() -> None:
    df = utils.read_df(sroot.OUTPUT_GPT4_PQ)
    df.sample(1).iloc[0].to_dict()

    # 검산
    df["msg_tokens"] = df["pred.messages"].progress_apply(
        lambda x: utils.num_tiktoken_from_messages(
            messages=json.loads(x), model=TIKTOKEN_MODEL_ID
        )
    )
    prompt_tokens_diff = (df["pred.prompt_tokens"] - df["msg_tokens"]).abs().sum()
    logger.debug(f"{prompt_tokens_diff=}")

    # check
    df["src_tokens"] = df["text.tgt"].progress_apply(
        lambda x: utils.num_tiktoken(s=x, model=TIKTOKEN_MODEL_ID)
    )
    df["tgt_tokens"] = df["pred.content"].progress_apply(
        lambda x: utils.num_tiktoken(s=x, model=TIKTOKEN_MODEL_ID)
    )
    logger.debug((df["tgt_tokens"] / df["src_tokens"]).describe())

    # ratio
    ratio: float = df["tgt_tokens"].sum() / df["src_tokens"].sum()
    logger.debug(f"{ratio=:.2f}")


def main() -> None:
    gen_estimation_json()
    if 0:
        check_mt_ratio()


if __name__ == "__main__":
    tqdm.pandas()
    pandarallel.initialize(progress_bar=True)
    #
    if hasattr(sys, "ps1"):
        pretty.install()
        reload(utils)
        reload(sroot)
    else:
        with logger.catch(onerror=lambda _: sys.exit(1)):
            # python -m src.eval.mt_aug.estimate_cost
            typer.run(main)
