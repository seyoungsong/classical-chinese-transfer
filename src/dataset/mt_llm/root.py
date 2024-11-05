import sys
from importlib import reload

import typer
from loguru import logger
from rich import pretty

from src import utils

if 0:
    __task_name = "dataset_mt_llm_v1"  # remove tower_mt
    __task_name = "dataset_mt_llm_v2"  # add cc_aug, change klc split
__task_name = "dataset_mt_llm_v3"
MODEL_DIR = utils.MODEL_ROOT_DIR / __task_name
SCRIPT_DIR = utils.SCRIPT_ROOT_DIR / __task_name
RESULT_DIR = utils.RESULT_ROOT_DIR / __task_name
TEMP_DIR = utils.TEMP_DIR / __task_name
HFACE_REPO_ID = "anonymous/Dataset-Hanja-MT-LLM"


CONCAT_PQ = TEMP_DIR / "concat.parquet"
FORMAT_PQ = TEMP_DIR / "format.parquet"
FILTER_PQ = MODEL_DIR / "filter.parquet"
TRAIN_PQ = MODEL_DIR / "train.parquet"


STAT_JSON = RESULT_DIR / "stat.json"


def main() -> None:
    pass


if __name__ == "__main__":
    if hasattr(sys, "ps1"):
        pretty.install()
        reload(utils)
    else:
        with logger.catch(onerror=lambda _: sys.exit(1)):
            typer.run(main)
