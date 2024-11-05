import sys
from importlib import reload

import typer
from loguru import logger
from rich import pretty

from src import utils

__task_name = "crawl_mqs_hj_v1"
MODEL_DIR = utils.MODEL_ROOT_DIR / __task_name
SCRIPT_DIR = utils.SCRIPT_ROOT_DIR / __task_name
RESULT_DIR = utils.RESULT_ROOT_DIR / __task_name
TEMP_DIR = utils.TEMP_DIR / __task_name


TEMP_MQS_HJ_DAY_DIR = utils.TEMP_DIR / "crawl-mqs-hj-day"
TEMP_MQS_HJ_HTML_DIR = utils.TEMP_DIR / "crawl-mqs-hj-html"
MQS_HJ_ID_PKL = utils.DATASET_DIR / "MQS-hj-id.pkl.zst"
MQS_HJ_ENTRY_PKL = utils.DATASET_DIR / "MQS-hj-entry.pkl.zst"
MQS_HJ_HTML_7Z = utils.DATASET_DIR / "MQS-hj-html.7z"
MQS_HJ_SRC_PKL = utils.DATASET_DIR / "MQS-hj-src.pkl.zst"
MQS_HJ_CLEAN_PKL = utils.DATASET_DIR / "MQS-hj-clean.pkl.zst"


def main() -> None:
    pass


if __name__ == "__main__":
    if hasattr(sys, "ps1"):
        pretty.install()
        reload(utils)
    else:
        with logger.catch(onerror=lambda _: sys.exit(1)):
            typer.run(main)
