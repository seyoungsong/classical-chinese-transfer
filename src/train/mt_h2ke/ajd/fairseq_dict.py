import sys
from importlib import reload

import typer
from loguru import logger
from rich import pretty
from tqdm import tqdm

import src.tool.train as ttool
import src.train.mt_h2ke.ajd.root as sroot
from src import utils


def gen_encode_file() -> None:
    # https://github.com/facebookresearch/fairseq/blob/main/scripts/spm_encode.py
    # https://github.com/facebookresearch/fairseq/blob/main/examples/translation/prepare-iwslt17-multilingual.sh#L114

    # read
    df = utils.read_df(sroot.DATASET_PQ)

    # load
    sp = ttool.load_spm_model(sroot.SPM_MODEL_FILE)

    # test
    x = "일본국왕(日本國王)이   \n빙문(聘問)하였다.\nOn    asdf day."
    x_encode = ttool.spm_encode(sp=sp, s=x)
    x_decode = ttool.spm_decode(sp=sp, s=x_encode)
    logger.debug(f"\nX: [{x}]\nE: {x_encode.split()}\nD: [{x_decode}]")

    # encode as pieces with token. 4 min
    logger.debug("encode.src...")
    df["encode.src"] = df["text.src"].progress_apply(
        lambda x: ttool.spm_encode(sp=sp, s=x)
    )
    #
    logger.debug("encode.tgt...")
    df["encode.tgt"] = df["text.tgt"].progress_apply(
        lambda x: ttool.spm_encode(sp=sp, s=x)
    )

    # sort cols
    c1 = [c for c in df.columns if not c.startswith("meta")]
    c2 = [c for c in df.columns if c.startswith("meta")]
    cols = sorted(c1) + sorted(c2)
    df = df[cols].reset_index(drop=True)
    df.sample(1).iloc[0].to_dict()

    # check unk
    if 0:
        str(sp.DecodeIds([sp.unk_id()])).strip()
        unk_str = "⁇"
        df["encode.src"].str.contains(unk_str).sum()
        df["encode.tgt"].str.contains(unk_str).sum()

    # save
    sroot.ENCODE_PQ.parent.mkdir(exist_ok=True, parents=True)
    utils.write_df2(sroot.ENCODE_PQ, df)  # 677.6M, 4c890a70


def gen_fairseq_dict_train_dir() -> None:
    # read
    df = utils.read_df(sroot.ENCODE_PQ)
    df.sample(1).iloc[0].to_dict()

    # filter: train only
    df["split"].value_counts()
    idx = df["split"] == "train"
    df = df[idx].reset_index(drop=True)

    # name
    subset, src_lang, tgt_lang = ("train", "src", "tgt")

    # write
    sroot.FAIRSEQ_DICT_TRAIN_DIR.mkdir(exist_ok=True, parents=True)
    fname = sroot.FAIRSEQ_DICT_TRAIN_DIR / f"{subset}.{src_lang}-{tgt_lang}.{src_lang}"
    utils.write_str(fname, "\n".join(df["encode.src"]))
    fname = sroot.FAIRSEQ_DICT_TRAIN_DIR / f"{subset}.{src_lang}-{tgt_lang}.{tgt_lang}"
    utils.write_str(fname, "\n".join(df["encode.tgt"]))


def cmd_fairseq_build_dict() -> None:
    # https://github.com/facebookresearch/fairseq/blob/main/docs/getting_started.rst#data-pre-processing

    # name
    subset, src_lang, tgt_lang = ("train", "src", "tgt")

    # script
    train_pref = sroot.FAIRSEQ_DICT_TRAIN_DIR / f"{subset}.{src_lang}-{tgt_lang}"
    script = f"""
    fairseq-preprocess \\
        --cpu \\
        --destdir {sroot.FAIRSEQ_DICT_DIR} \\
        --dict-only \\
        --joined-dictionary \\
        --nwordssrc 32000 \\
        --nwordstgt 32000 \\
        --seed 42 \\
        --source-lang {src_lang} \\
        --target-lang {tgt_lang} \\
        --trainpref {train_pref} \\
        --workers 8
    """
    sroot.FAIRSEQ_DICT_DIR.mkdir(exist_ok=True, parents=True)
    sroot.SCRIPT_DIR.mkdir(exist_ok=True, parents=True)
    utils.write_sh(sroot.SCRIPT_DIR / "fairseq_build_dict.sh", script)


def run_fairseq_build_dict() -> None:
    fname = sroot.SCRIPT_DIR / "fairseq_build_dict.sh"
    assert fname.is_file(), f"not found: {fname}"
    utils.reset_dir(sroot.FAIRSEQ_DICT_DIR)
    cmd = f"bash {fname}"
    utils.subprocess_run(cmd)
    utils.log_written(sroot.DATA_DICT_TXT)
    if 0:
        utils.open_code(sroot.FAIRSEQ_DICT_DIR / "preprocess.log")


def check_fairseq_dict() -> None:
    # count
    s1 = utils.read_str(sroot.DATA_DICT_TXT)
    lines = s1.strip().splitlines()
    logger.debug(
        f"lines: {len(lines)} + 4 = {len(lines) + 4}"
    )  # +4 for <s>, <unk>, </s>, <pad>

    # check
    f2 = sroot.DATA_DICT_TXT.parent / "dict.tgt.txt"
    s2 = utils.read_str(f2)
    assert s1 == s2, "two dict mismatch"

    # log
    utils.log_written(sroot.DATA_DICT_TXT)


def main() -> None:
    # encode
    gen_encode_file()  # 565.6M, 27a259f0, 416819
    # dict
    gen_fairseq_dict_train_dir()
    cmd_fairseq_build_dict()
    run_fairseq_build_dict()
    check_fairseq_dict()  # 412.2K, 7bf938dc


if __name__ == "__main__":
    tqdm.pandas()
    #
    if hasattr(sys, "ps1"):
        pretty.install()
        reload(utils)
        reload(sroot)
        reload(ttool)
    else:
        with logger.catch(onerror=lambda _: sys.exit(1)):
            # python -m src.train.mt_h2ke.ajd.fairseq_dict
            typer.run(main)
