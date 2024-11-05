import sys
from importlib import reload

import typer
from loguru import logger
from rich import pretty

from src import utils

__task_name = "train_mt_h2ke_ajd_klc_v1"
MODEL_DIR = utils.MODEL_ROOT_DIR / __task_name
SCRIPT_DIR = utils.SCRIPT_ROOT_DIR / __task_name
RESULT_DIR = utils.RESULT_ROOT_DIR / __task_name
TEMP_DIR = utils.TEMP_DIR / __task_name
HFACE_REPO_ID = "anonymous/H2KE-AJD-KLC-ct2"

MAX_LEN = 512

DATASET_PQ = MODEL_DIR / "dataset.parquet"

SPM_TRAIN_TXT = TEMP_DIR / "spm_train.txt"
SPM_TRAIN_SAMPLE_TXT = TEMP_DIR / "spm_train_sample.txt"

__SPM_MODEL_DIR = MODEL_DIR / "spm_model"
SPM_MODEL_FILE, SPM_VOCAB_FILE = utils.get_spm_path(__SPM_MODEL_DIR)


ENCODE_PQ = TEMP_DIR / "encode.parquet"


FAIRSEQ_DICT_TRAIN_DIR = TEMP_DIR / "fairseq_dict_train"
FAIRSEQ_DICT_DIR = MODEL_DIR / "fairseq_dict"
DATA_DICT_TXT = FAIRSEQ_DICT_DIR / "dict.src.txt"


TRUNCATE_PQ = TEMP_DIR / "truncate.parquet"
FAIRSEQ_TEXT_DIR = TEMP_DIR / "fairseq_text"
FAIRSEQ_BIN_DIR = TEMP_DIR / "fairseq_bin"


FAIRSEQ_INFO_DIR = MODEL_DIR / "fairseq_info"
# e.g. https://dl.fbaipublicfiles.com/m2m_100/language_pairs.txt
LANG_PAIRS_TXT = FAIRSEQ_INFO_DIR / "lang_pairs.txt"
LANG_DICT_TXT = FAIRSEQ_INFO_DIR / "lang_dict.txt"


FAIRSEQ_TRAIN_DIR = TEMP_DIR / "fairseq_train"
FAIRSEQ_TENSORBOARD_DIR = FAIRSEQ_TRAIN_DIR / "tensorboard"
FAIRSEQ_CKPT_DIR = FAIRSEQ_TRAIN_DIR / "checkpoint"


FAIRSEQ_MODEL_DIR = MODEL_DIR / "fairseq_model"
CT2_TEMP_DIR = TEMP_DIR / "ct2_temp"
CT2_MODEL_DIR = MODEL_DIR / "ct2_model"


def main() -> None:
    pass


if __name__ == "__main__":
    if hasattr(sys, "ps1"):
        pretty.install()
        reload(utils)
    else:
        with logger.catch(onerror=lambda _: sys.exit(1)):
            typer.run(main)
