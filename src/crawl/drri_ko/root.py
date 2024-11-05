import sys
from importlib import reload

import typer
from loguru import logger
from rich import pretty

from src import utils

__task_name = "crawl_drri_ko_v1"
MODEL_DIR = utils.MODEL_ROOT_DIR / __task_name
SCRIPT_DIR = utils.SCRIPT_ROOT_DIR / __task_name
RESULT_DIR = utils.RESULT_ROOT_DIR / __task_name
TEMP_DIR = utils.TEMP_DIR / __task_name

BOOK_JSONL = MODEL_DIR / "book.jsonl"
BOOK2_JSONL = MODEL_DIR / "book2.jsonl"

LV1_TASK_PQ = MODEL_DIR / "lv1_task.parquet"
LV1_DL_DIR = TEMP_DIR / "lv1_dl"
LV1_PQ = MODEL_DIR / "lv1.parquet"
LV1A_PQ = MODEL_DIR / "lv1a.parquet"

CRAWL_TASK_PQ = MODEL_DIR / "crawl_task.parquet"
CRAWL_DL_DIR = TEMP_DIR / "crawl_dl"
CRAWL_7Z = MODEL_DIR / "crawl.7z"

PARSE_DIR = TEMP_DIR / "parse"
PARSE_CONCAT_JSONL = TEMP_DIR / "parse_concat.jsonl"
PARSE_PQ = MODEL_DIR / "parse.parquet"
CLEAN_PQ = MODEL_DIR / "clean.parquet"
CLEAN2_PQ = MODEL_DIR / "clean2.parquet"
PARSE2_PQ = MODEL_DIR / "parse2.parquet"
PARSE3_PQ = MODEL_DIR / "parse3.parquet"
FORMAT_PQ = MODEL_DIR / "format.parquet"
FORMAT2_PQ = MODEL_DIR / "format2.parquet"

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
