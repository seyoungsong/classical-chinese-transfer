import sys
from importlib import reload

import typer
from loguru import logger
from rich import pretty

from src import utils

__task_name = "train_ner1_ajd_cc_v1"
MODEL_DIR = utils.MODEL_ROOT_DIR / __task_name
SCRIPT_DIR = utils.SCRIPT_ROOT_DIR / __task_name
RESULT_DIR = utils.RESULT_ROOT_DIR / __task_name
TEMP_DIR = utils.TEMP_DIR / __task_name
HFACE_REPO_ID = "anonymous/SikuRoBERTa-NER1-AJD-CC"


DATASET_PQ = MODEL_DIR / "dataset.parquet"


STAT_DIR = MODEL_DIR / "stat"
LABELS_JSON = MODEL_DIR / "labels.json"
LABELS_TSV = MODEL_DIR / "labels.tsv"


HFACE_PQ = TEMP_DIR / "hface.parquet"
HFACE_JSONL_DIR = TEMP_DIR / "hface_jsonl"
HFACE_JSONL2_DIR = TEMP_DIR / "hface_jsonl2"
HFACE_TRAIN_DIR = TEMP_DIR / "hface_train"
HFACE_MODEL_DIR = MODEL_DIR / "hface_model"


def main() -> None:
    pass


if __name__ == "__main__":
    if hasattr(sys, "ps1"):
        pretty.install()
        reload(utils)
    else:
        with logger.catch(onerror=lambda _: sys.exit(1)):
            typer.run(main)
