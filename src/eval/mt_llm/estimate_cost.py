import json
import sys
from importlib import reload

import typer
from loguru import logger
from pandarallel import pandarallel
from rich import pretty
from tqdm import tqdm

import src.dataset.mt_aug.root
import src.dataset.mt_eval.root
import src.dataset.mt_h2ke.root
import src.eval.mt_llm.root as sroot
from src import utils

TIKTOKEN_MODEL_ID = "gpt-4-0613"


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


def gen_est_data_file() -> None:
    # read
    df0 = utils.read_df(src.dataset.mt_eval.root.EVAL_PQ)
    df = df0.copy()

    # check
    df.sample(1).iloc[0].to_dict()
    df.groupby(["meta.corpus", "lang.src", "lang.tgt"]).size()

    # save
    sroot.EST_DATASET_PQ.parent.mkdir(parents=True, exist_ok=True)
    utils.write_df2(sroot.EST_DATASET_PQ, df)


def gen_estimation_json() -> None:
    # read
    df0 = utils.read_df(sroot.EST_DATASET_PQ)
    df = df0.copy()
    df.sample(1).iloc[0].to_dict()

    # count
    df["src_tokens"] = df["text.src"].progress_apply(
        lambda x: utils.num_tiktoken(s=x, model=TIKTOKEN_MODEL_ID)
    )
    df["tgt_tokens"] = df["text.tgt"].progress_apply(
        lambda x: utils.num_tiktoken(s=x, model=TIKTOKEN_MODEL_ID)
    )

    # sum
    df1 = (
        df.groupby(["meta.corpus", "lang.src", "lang.tgt"])
        .agg({"src_tokens": "sum", "tgt_tokens": "sum"})
        .reset_index()
    )
    df1["cost.gpt4"] = df1.apply(
        lambda x: utils.calculate_openai_pricing(
            input_tokens=x["src_tokens"],
            output_tokens=x["tgt_tokens"],
            model="gpt-4-0125-preview",
        ),
        axis=1,
    )
    df1["cost.gpt3"] = df1.apply(
        lambda x: utils.calculate_openai_pricing(
            input_tokens=x["src_tokens"],
            output_tokens=x["tgt_tokens"],
            model="gpt-3.5-turbo-0125",
        ),
        axis=1,
    )
    ADJUSTMENT = 1 / 1.09  # from tech report
    df1["cost.hcx3"] = df1.apply(
        lambda x: utils.calculate_hcx_pricing(
            input_tokens=x["src_tokens"] * ADJUSTMENT,
            output_tokens=x["tgt_tokens"] * ADJUSTMENT,
            model="HCX-003",
        ),
        axis=1,
    )

    # save
    sroot.ESTIMATION_TSV.parent.mkdir(parents=True, exist_ok=True)
    utils.write_df(sroot.ESTIMATION_TSV, df1)
    utils.write_df(sroot.ESTIMATION_JSON, df1)

    # etc
    df1 = utils.read_df(sroot.ESTIMATION_JSON)
    df1.sum()
    df1[[c for c in df1.columns if "cost" in c]].sum()
    """
    cost.gpt4    105.02
    cost.gpt3      5.25
    cost.hcx3     15.75
    """
    15.75 * 1380.28


def check_mt_ratio() -> None:
    df = utils.read_df(sroot.OUTPUT2_PQ)
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
    gen_est_data_file()
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
            # python -m src.eval.mt_llm.estimate_cost
            typer.run(main)
