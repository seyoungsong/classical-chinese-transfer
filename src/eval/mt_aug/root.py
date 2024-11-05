import sys
from importlib import reload

import typer
from loguru import logger
from rich import pretty

from src import utils

__task_name = "eval_mt_aug_v1"
MODEL_DIR = utils.MODEL_ROOT_DIR / __task_name
SCRIPT_DIR = utils.SCRIPT_ROOT_DIR / __task_name
RESULT_DIR = utils.RESULT_ROOT_DIR / __task_name
TEMP_DIR = utils.TEMP_DIR / __task_name
HFACE_REPO_ID = "anonymous/MT-AUG-GPT4"


DATASET_PQ = MODEL_DIR / "dataset.parquet"
ESTIMATION_JSON = RESULT_DIR / "estimation.json"


OUTPUT_GPT3_PQ = MODEL_DIR / "output_gpt3.parquet"
OUTPUT_GPT4_PQ = MODEL_DIR / "output_gpt4.parquet"


def main() -> None:
    pass


if __name__ == "__main__":
    if hasattr(sys, "ps1"):
        pretty.install()
        reload(utils)
    else:
        with logger.catch(onerror=lambda _: sys.exit(1)):
            typer.run(main)
