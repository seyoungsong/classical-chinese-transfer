import itertools
import sys
from importlib import reload

import typer
from loguru import logger
from rich import pretty

from src import utils

PREF_STR = """
python -m src.train.ner.ajd_klc_cc_2to1
python -m src.train.ner.ajd_klc_cc_2to0
python -m src.train.ner.ajd_klc_cc_1to1
python -m src.train.ner.ajd_klc_cc_1to0
python -m src.train.ner.ajd_klc_cc_05to1
python -m src.train.ner.ajd_klc_cc_05to0
python -m src.train.ner.ajd_klc_cc_1_4to1
python -m src.train.ner.ajd_klc_cc_1_4to0
python -m src.train.ner.ajd_klc_cc_1_8to1
python -m src.train.ner.ajd_klc_cc_1_8to0
python -m src.train.ner.ajd_klc_cc_1_16to1
python -m src.train.ner.ajd_klc_cc_1_16to0
python -m src.train.ner.ajd_klc_cc_1_32to1
python -m src.train.ner.ajd_klc_cc_1_32to0
"""

TASK_TRAIN_STR = """
dataset
label_prepare
hface_prepare
hface_train --run
"""


TASK_FINISH_STR = """
hface_convert
hface_upload --run
"""


def print_train_sh() -> None:
    pref_list = PREF_STR.strip().split("\n")
    pref_list = [s.strip() for s in pref_list if s.strip()]
    task_list = TASK_TRAIN_STR.strip().split("\n")
    task_list = [s.strip() for s in task_list if s.strip()]
    # pref x task
    cmd_list = [
        f"{s1}.{s2}" for s1, s2 in list(itertools.product(pref_list, task_list))
    ]
    cmd = "\n".join(cmd_list)
    print(cmd)


def print_upload_sh() -> None:
    pref_list = PREF_STR.strip().split("\n")
    pref_list = [s.strip() for s in pref_list if s.strip()]
    task_list = TASK_FINISH_STR.strip().split("\n")
    task_list = [s.strip() for s in task_list if s.strip()]
    # pref x task
    cmd_list = [
        f"{s1}.{s2}" for s1, s2 in list(itertools.product(pref_list, task_list))
    ]
    cmd = "\n".join(cmd_list)
    print(cmd)


def print_import() -> None:
    pref_list = PREF_STR.strip().split("\n")
    pref_list = [s.strip() for s in pref_list if s.strip()]
    task_list = "model".strip().split("\n")
    task_list = [s.strip() for s in task_list if s.strip()]
    # pref x task
    cmd_list = [
        f"{s1}.{s2}" for s1, s2 in list(itertools.product(pref_list, task_list))
    ]
    cmd = "\n".join(cmd_list)
    cmd = cmd.replace("python -m", "import")
    print(cmd)


def print_elif() -> None:
    pref_list = PREF_STR.strip().split("\n")
    pref_list = [s.strip() for s in pref_list if s.strip()]
    pref_list = [s.replace("python -m src.train.ner.", "") for s in pref_list]
    cmd_list = [
        f"""
    elif model_id == "{s}": model = src.train.ner.{s}.model.HanjaNER(device=device, torch_dtype=torch_dtype)
    """.strip()
        for s in pref_list
    ]
    cmd = "\n".join(cmd_list)
    print(cmd)


def print_inference() -> None:
    pref_list = PREF_STR.strip().split("\n")
    pref_list = [s.strip() for s in pref_list if s.strip()]
    pref_list = [s.replace("python -m src.train.ner.", "") for s in pref_list]
    cmd = "\n".join(pref_list)
    print(cmd)


def print_url() -> None:
    urls = [
        f"https://huggingface.co/anonymous/SikuRoBERTa-NER-{s}/tensorboard?params=scalars%26tagFilter%3Depoch%26_smoothingWeight%3D0#frame"
        for s in [
            "CC",
            "AJD",
            "AJD-CC",
            "AJD-KLC",
            "AJD-KLC-CC",
        ]
    ]
    pref_list = PREF_STR.strip().split("\n")
    pref_list = [s.strip() for s in pref_list if s.strip()]
    pref_list = [
        s.replace("python -m src.train.ner.ajd_klc_cc_", "") for s in pref_list
    ]
    pref_list = [
        f"https://huggingface.co/anonymous/SikuRoBERTa-NER-AJD-KLC-CC-{s}/tensorboard?params=scalars%26tagFilter%3Depoch%26_smoothingWeight%3D0#frame"
        for s in pref_list
    ]
    pref_list = urls + pref_list
    cmd = "\n".join(pref_list)
    print(cmd)


def main() -> None:
    print_train_sh()
    print_upload_sh()
    print_import()
    print_elif()
    print_url()


if __name__ == "__main__":
    if hasattr(sys, "ps1"):
        pretty.install()
        reload(utils)
    else:
        with logger.catch(onerror=lambda _: sys.exit(1)):
            # python -m src.train.ner.train
            typer.run(main)
