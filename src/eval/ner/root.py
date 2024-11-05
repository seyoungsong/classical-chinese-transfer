import sys
from importlib import reload

import typer
from loguru import logger
from rich import pretty

from src import utils

__task_name = "eval_ner_v2"
MODEL_DIR = utils.MODEL_ROOT_DIR / __task_name
SCRIPT_DIR = utils.SCRIPT_ROOT_DIR / __task_name
RESULT_DIR = utils.RESULT_ROOT_DIR / __task_name
TEMP_DIR = utils.TEMP_DIR / __task_name
HFACE_REPO_ID = "anonymous/Eval-NER"


PREV_OUTPUT_PQ = MODEL_DIR / "prev_output.parquet"
DATASET_PQ = MODEL_DIR / "dataset.parquet"
OUTPUT_PQ = MODEL_DIR / "output.parquet"
OUTPUT2_PQ = MODEL_DIR / "output2.parquet"


SCORE_F1_BINARY_TSV = RESULT_DIR / "score_f1_binary.tsv"
SCORE_F1_ENTITY_TSV = RESULT_DIR / "score_f1_entity.tsv"
SCORE_NUM_SAMPLE_TSV = RESULT_DIR / "score_num_sample.tsv"


def main() -> None:
    pass


if __name__ == "__main__":
    if hasattr(sys, "ps1"):
        pretty.install()
        reload(utils)
    else:
        with logger.catch(onerror=lambda _: sys.exit(1)):
            typer.run(main)
