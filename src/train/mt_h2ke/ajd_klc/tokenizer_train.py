import os
import sys
from importlib import reload

import pandas as pd
import sentencepiece
import typer
from loguru import logger
from rich import pretty

import src.train.mt_h2ke.ajd_klc.root as sroot
from src import utils


def train_tokenizer() -> None:
    # files
    model_prefix = str(sroot.SPM_MODEL_FILE).rsplit(".model")[0]
    input_files = str(sroot.SPM_TRAIN_TXT.resolve())
    num_threads = int(os.cpu_count() * 0.9)  # type: ignore

    # check
    s1 = utils.read_str(sroot.SPM_TRAIN_TXT)
    df1 = pd.DataFrame()
    df1["text"] = s1.splitlines()
    logger.debug(df1["text"].str.len().max())  # 225390
    del s1, df1

    # https://github.com/google/sentencepiece/blob/master/doc/options.md
    sroot.SPM_MODEL_FILE.parent.mkdir(exist_ok=True, parents=True)
    sentencepiece.SentencePieceTrainer.Train(
        character_coverage=0.9995,  # default: 0.9995
        input=input_files,
        max_sentence_length=999999,  # default: 4192
        model_prefix=model_prefix,
        model_type="unigram",
        num_threads=num_threads,
        vocab_size=32000,  # default: 8000
    )

    # log
    utils.log_written(sroot.SPM_MODEL_FILE)


def main() -> None:
    train_tokenizer()  # 741.7K, f7b91ae8


if __name__ == "__main__":
    if hasattr(sys, "ps1"):
        pretty.install()
        reload(utils)
        reload(sroot)
    else:
        with logger.catch(onerror=lambda _: sys.exit(1)):
            # python -m src.train.mt_h2ke.ajd_klc.tokenizer_train
            typer.run(main)
