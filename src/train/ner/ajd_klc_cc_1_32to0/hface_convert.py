import shutil
import sys
from importlib import reload
from pathlib import Path

import pandas as pd
import typer
from loguru import logger
from rich import pretty
from tqdm import tqdm

import src.train.ner.ajd_klc_cc_1_32to0.root as sroot
from src import utils


def check_model_file() -> None:
    # list
    train_dir = sroot.HFACE_TRAIN_DIR
    if 0:
        utils.open_code(train_dir)
    fnames = [p for p in train_dir.rglob("*") if p.is_file()]
    fnames = [p for p in fnames if "checkpoint-" not in str(p)]

    # sort
    df = pd.DataFrame()
    df["fname"] = fnames
    df["size"] = df["fname"].apply(lambda p: Path(p).stat().st_size)
    df.sort_values("size", inplace=True, ignore_index=True, ascending=True)
    df.to_dict(orient="records")

    # log
    model_file = train_dir / "model.safetensors"
    utils.log_written(model_file)


def gen_model_dir() -> None:
    # dir
    train_dir = sroot.HFACE_TRAIN_DIR
    model_dir = sroot.HFACE_MODEL_DIR
    if 0:
        utils.open_code(model_dir)

    # files
    fnames = [p for p in train_dir.glob("*") if p.is_file()]
    model_dir.mkdir(parents=True, exist_ok=True)
    pairs = [(p, model_dir / p.name) for p in fnames]
    for f1, f2 in tqdm(pairs):
        _ = shutil.copy2(f1, f2)

    # dirs
    dirs = [p for p in train_dir.glob("*") if p.is_dir()]
    dirs = [p for p in dirs if "checkpoint" not in p.name]
    pairs = [(p, model_dir / p.name) for p in dirs]
    for d1, d2 in tqdm(pairs):
        shutil.copytree(d1, d2, dirs_exist_ok=True)

    # log
    utils.log_written2(model_dir)


def main() -> None:
    check_model_file()  # 413.3M, d8b8f060
    gen_model_dir()  # 430.6M
    if 0:
        utils.open_code(sroot.HFACE_MODEL_DIR)


if __name__ == "__main__":
    if hasattr(sys, "ps1"):
        pretty.install()
        reload(utils)
        reload(sroot)
    else:
        with logger.catch(onerror=lambda _: sys.exit(1)):
            # python -m src.train.ner.ajd_klc_cc_2to1.hface_convert
            typer.run(main)
