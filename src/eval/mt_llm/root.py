import sys
from importlib import reload

import typer
from loguru import logger
from rich import pretty

from src import utils

__task_name = "eval_mt_llm_v3"
MODEL_DIR = utils.MODEL_ROOT_DIR / __task_name
SCRIPT_DIR = utils.SCRIPT_ROOT_DIR / __task_name
RESULT_DIR = utils.RESULT_ROOT_DIR / __task_name
TEMP_DIR = utils.TEMP_DIR / __task_name
HFACE_REPO_ID = "anonymous/Eval-MT-LLM"


PREV_OUTPUT_PQ = MODEL_DIR / "prev_output.parquet"
DATASET_PQ = MODEL_DIR / "dataset.parquet"

EST_DATASET_PQ = TEMP_DIR / "est_dataset.parquet"
ESTIMATION_TSV = RESULT_DIR / "estimation.tsv"
ESTIMATION_JSON = RESULT_DIR / "estimation.json"


OUTPUT_PQ = MODEL_DIR / "output.parquet"
OUTPUT2_PQ = MODEL_DIR / "output2.parquet"


SCORE_BLEU_TSV = RESULT_DIR / "score_bleu.tsv"
SCORE_SPBLEU_TSV = RESULT_DIR / "score_spbleu.tsv"
SCORE_NUM_SAMPLE_TSV = RESULT_DIR / "score_num_sample.tsv"


SCORE_BS_BLEU_TSV = RESULT_DIR / "score_bs_bleu.tsv"
SCORE_BS_SPBLEU_TSV = RESULT_DIR / "score_bs_spbleu.tsv"
SCORE_BS_NUM_SAMPLE_TSV = RESULT_DIR / "score_bs_num_sample.tsv"


def main() -> None:
    pass


if __name__ == "__main__":
    if hasattr(sys, "ps1"):
        pretty.install()
        reload(utils)
    else:
        with logger.catch(onerror=lambda _: sys.exit(1)):
            typer.run(main)
