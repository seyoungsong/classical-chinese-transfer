import sys
from importlib import reload

import typer
from loguru import logger
from rich import pretty

from src import utils

__task_name = "eval_vocab_v2"
MODEL_DIR = utils.MODEL_ROOT_DIR / __task_name
SCRIPT_DIR = utils.SCRIPT_ROOT_DIR / __task_name
RESULT_DIR = utils.RESULT_ROOT_DIR / __task_name
TEMP_DIR = utils.TEMP_DIR / __task_name


DATASET_PQ = TEMP_DIR / "dataset.parquet"


CHAR_COUNT_DIR = MODEL_DIR / "char_count"
CHAR_COUNT_PQ = MODEL_DIR / "char_count.parquet"
CHAR_COUNT2_PQ = MODEL_DIR / "char_count2.parquet"
SAMPLE_PQ = TEMP_DIR / "sample.parquet"
PPL_SAMPLE_PQ = MODEL_DIR / "ppl_sample.parquet"


def main() -> None:
    pass


if __name__ == "__main__":
    if hasattr(sys, "ps1"):
        pretty.install()
        reload(utils)
    else:
        with logger.catch(onerror=lambda _: sys.exit(1)):
            typer.run(main)
