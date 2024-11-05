import sys
from importlib import reload

import typer
from loguru import logger
from rich import pretty

from src import utils

__task_name = "corpus_wyweb_ner_v1"
MODEL_DIR = utils.MODEL_ROOT_DIR / __task_name
SCRIPT_DIR = utils.SCRIPT_ROOT_DIR / __task_name
RESULT_DIR = utils.RESULT_ROOT_DIR / __task_name
TEMP_DIR = utils.TEMP_DIR / __task_name


ALIGN_PQ = MODEL_DIR / "align.parquet"
FORMAT_PQ = MODEL_DIR / "format.parquet"
CLEAN_PQ = MODEL_DIR / "clean.parquet"
NORMAL_PQ = MODEL_DIR / "normal.parquet"
SPLIT_PQ = MODEL_DIR / "split.parquet"
FILTER_PQ = MODEL_DIR / "filter.parquet"


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