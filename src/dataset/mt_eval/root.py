import sys
from importlib import reload

import typer
from loguru import logger
from rich import pretty

from src import utils

# v1: 2k, v2: +ocdb(but error), v3: ~6k(w/v1)
__task_name = "dataset_mt_eval_v3"
MODEL_DIR = utils.MODEL_ROOT_DIR / __task_name
SCRIPT_DIR = utils.SCRIPT_ROOT_DIR / __task_name
RESULT_DIR = utils.RESULT_ROOT_DIR / __task_name
TEMP_DIR = utils.TEMP_DIR / __task_name
HFACE_REPO_ID = "anonymous/Dataset-Hanja-MT-Eval"


CONCAT_PQ = TEMP_DIR / "concat.parquet"
FILTER_PQ = MODEL_DIR / "filter.parquet"
FILTER2_PQ = TEMP_DIR / "filter2.parquet"
FILTER3_PQ = TEMP_DIR / "filter3.parquet"
FILTER4_PQ = MODEL_DIR / "filter4.parquet"
EVAL_PQ = MODEL_DIR / "eval.parquet"


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
