import sys
from importlib import reload

import typer
from loguru import logger
from rich import pretty

from src import utils

__task_name = "train_mt_llm_ajd_klc_cc_v3"
MODEL_DIR = utils.MODEL_ROOT_DIR / __task_name
SCRIPT_DIR = utils.SCRIPT_ROOT_DIR / __task_name
RESULT_DIR = utils.RESULT_ROOT_DIR / __task_name
TEMP_DIR = utils.TEMP_DIR / __task_name
HFACE_REPO_ID = "anonymous/TowerInstruct-7B-v0.2-AJD-KLC-CC-QLoRA"


DATASET_PQ = MODEL_DIR / "dataset.parquet"


HFACE_INFO_JSON = utils.SCRIPT_ROOT_DIR.parent / "hface_info.json"
HFACE_TRAIN_DIR = TEMP_DIR / "hface_train"
HFACE_LORA_DIR = MODEL_DIR / "hface_lora"


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
