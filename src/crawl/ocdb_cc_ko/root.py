import sys
from importlib import reload

import typer
from loguru import logger
from rich import pretty

from src import utils

__task_name = "crawl_ocdb_cc_ko_v1"
MODEL_DIR = utils.MODEL_ROOT_DIR / __task_name
SCRIPT_DIR = utils.SCRIPT_ROOT_DIR / __task_name
RESULT_DIR = utils.RESULT_ROOT_DIR / __task_name
TEMP_DIR = utils.TEMP_DIR / __task_name


BOOK_JSONL = MODEL_DIR / "book.jsonl"
BOOK2_JSONL = MODEL_DIR / "book2.jsonl"


LV1_TASK_PQ = MODEL_DIR / "lv1_task.parquet"
LV1_DL_DIR = TEMP_DIR / "lv1_dl"
LV1_7Z = MODEL_DIR / "lv1.7z"
LV1_PARSE_DIR = TEMP_DIR / "lv1_parse"
LV1_PARSE_JSONL = TEMP_DIR / "lv1_parse.jsonl"
LV1_PQ = MODEL_DIR / "lv1.parquet"
LV1A_PQ = MODEL_DIR / "lv1a.parquet"


LV2_TASK_PQ = MODEL_DIR / "lv2_task.parquet"
LV2_DL_DIR = TEMP_DIR / "lv2_dl"
LV2_7Z = MODEL_DIR / "lv2.7z"
LV2_PARSE_DIR = TEMP_DIR / "lv2_parse"
LV2_PARSE_JSONL = TEMP_DIR / "lv2_parse.jsonl"
LV2_PQ = MODEL_DIR / "lv2.parquet"
LV2A_PQ = MODEL_DIR / "lv2a.parquet"


CLEAN_PQ = TEMP_DIR / "clean.parquet"
CLEAN2_PQ = TEMP_DIR / "clean2.parquet"
PARSE2_PQ = TEMP_DIR / "parse2.parquet"
FORMAT_PQ = TEMP_DIR / "format.parquet"
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
