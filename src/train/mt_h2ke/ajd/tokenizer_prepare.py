import sys
from importlib import reload

import typer
from loguru import logger
from rich import pretty
from tqdm import tqdm

import src.train.mt_h2ke.ajd.root as sroot
from src import utils


def gen_spm_train_txt() -> None:
    # read
    df0 = utils.read_df(sroot.DATASET_PQ)
    df = df0.copy()

    # drop: use train only
    idx = df["split"] == "train"
    df = df[idx].reset_index(drop=True)
    df.groupby(["meta.corpus", "lang.src", "lang.tgt"]).size()

    # merge: src_text and tgt_text
    cols = ["text.src", "text.tgt"]
    text: utils.SeriesType = df[cols].stack().reset_index(drop=True)  # type: ignore

    # dedup
    text.duplicated().mean() * 100  # 10.8%
    text.drop_duplicates(inplace=True)
    text.reset_index(drop=True, inplace=True)

    # check
    logger.debug(text.apply(len).describe())

    # sample
    sroot.SPM_TRAIN_SAMPLE_TXT.parent.mkdir(exist_ok=True, parents=True)
    text_sample = text.sample(n=100, random_state=42).sort_index()
    utils.write_str(sroot.SPM_TRAIN_SAMPLE_TXT, "\n".join(text_sample))
    utils.log_written(sroot.SPM_TRAIN_SAMPLE_TXT)

    # write
    sroot.SPM_TRAIN_TXT.parent.mkdir(exist_ok=True, parents=True)
    utils.write_str(sroot.SPM_TRAIN_TXT, "\n".join(text))


def main() -> None:
    gen_spm_train_txt()  # 462.2M, e65237e7


if __name__ == "__main__":
    tqdm.pandas()
    #
    if hasattr(sys, "ps1"):
        pretty.install()
        reload(utils)
        reload(sroot)
    else:
        with logger.catch(onerror=lambda _: sys.exit(1)):
            # python -m src.train.mt_h2ke.ajd.tokenizer_prepare
            typer.run(main)
