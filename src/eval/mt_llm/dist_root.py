import sys
from importlib import reload

import typer
from loguru import logger
from rich import pretty

from src import utils

__task_name = "eval_mt_llm_dist_v3"
MODEL_DIR = utils.TEMP_DIR / __task_name


INPUT_PQ = MODEL_DIR / "input.parquet"
INPUT_DIR = MODEL_DIR / "input"
ERROR_DIR = MODEL_DIR / "error"
OUTPUT_DIR = MODEL_DIR / "output"
OUTPUT_JSONL = MODEL_DIR / "output.jsonl"


def main() -> None:
    pass


if __name__ == "__main__":
    if hasattr(sys, "ps1"):
        pretty.install()
        reload(utils)
    else:
        with logger.catch(onerror=lambda _: sys.exit(1)):
            typer.run(main)
