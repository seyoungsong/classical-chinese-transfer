import sys
from importlib import reload

import typer
from loguru import logger
from rich import pretty

from src import utils

__task_name = "crawl_ajd_link_v1"
MODEL_DIR = utils.MODEL_ROOT_DIR / __task_name
SCRIPT_DIR = utils.SCRIPT_ROOT_DIR / __task_name
RESULT_DIR = utils.RESULT_ROOT_DIR / __task_name
TEMP_DIR = utils.TEMP_DIR / __task_name

BOOK_PQ = MODEL_DIR / "book.parquet"

LV1_TASK_PQ = MODEL_DIR / "lv1_task.parquet"
LV1_DL_DIR = TEMP_DIR / "lv1_dl"
LV1_7Z = MODEL_DIR / "lv1.7z"
LV1_PARSE_DIR = TEMP_DIR / "lv1_parse"
LV1_PARSE_CONCAT_JSONL = TEMP_DIR / "lv1_parse_concat.jsonl"
LV1_PQ = MODEL_DIR / "lv1.parquet"
LV1A_PQ = MODEL_DIR / "lv1a.parquet"

FORMAT_PQ = MODEL_DIR / "format.parquet"


def main() -> None:
    pass


if __name__ == "__main__":
    if hasattr(sys, "ps1"):
        pretty.install()
        reload(utils)
    else:
        with logger.catch(onerror=lambda _: sys.exit(1)):
            typer.run(main)
