import shutil
import sys
from importlib import reload

import typer
from loguru import logger
from rich import pretty
from tqdm import tqdm

import src.train.mt_h2ke.ajd_klc.root as sroot
from src import utils


def gen_model_dir() -> None:
    # dir
    train_dir = sroot.FAIRSEQ_CKPT_DIR
    model_dir = sroot.FAIRSEQ_MODEL_DIR
    if 0:
        utils.open_code(model_dir)

    # files
    fnames = sorted([p for p in train_dir.glob("*") if p.is_file()])
    fnames = [p for p in fnames if "_best.pt" in p.name or "_last.pt" in p.name]
    model_dir.mkdir(parents=True, exist_ok=True)
    pairs = [(p, model_dir / p.name) for p in fnames]
    for f1, f2 in tqdm(pairs):
        _ = shutil.copy2(f1, f2)

    # dirs
    dirs = [sroot.FAIRSEQ_TENSORBOARD_DIR]
    pairs = [(p, model_dir / p.name) for p in dirs]
    for d1, d2 in tqdm(pairs):
        shutil.copytree(d1, d2, dirs_exist_ok=True)

    # etc (spm)
    fnames = [sroot.SPM_MODEL_FILE, sroot.SPM_VOCAB_FILE]
    pairs = [(p, model_dir / p.name) for p in fnames]
    for f1, f2 in tqdm(pairs):
        _ = shutil.copy2(f1, f2)

    # log
    utils.log_written2(model_dir)


def cmd_convert_fairseq_to_ct2() -> None:
    # https://opennmt.net/CTranslate2/guides/fairseq.html
    # https://opennmt.net/CTranslate2/quantization.html
    # https://github.com/OpenNMT/CTranslate2#gpu
    # https://opennmt.net/CTranslate2/guides/transformers.html#m2m-100
    # https://opennmt.net/CTranslate2/guides/fairseq.html#m2m-100

    best_ckpt_fname = sroot.FAIRSEQ_MODEL_DIR / "checkpoint_best.pt"
    if not best_ckpt_fname.is_file():
        logger.warning(f"{best_ckpt_fname} not found")

    # prepare
    sroot.CT2_MODEL_DIR.mkdir(exist_ok=True, parents=True)
    sroot.CT2_TEMP_DIR.mkdir(exist_ok=True, parents=True)

    # write script
    script = f"""
    # check
    tree -h --du -a {sroot.CT2_TEMP_DIR}

    # prepare
    cp {sroot.LANG_DICT_TXT} {sroot.CT2_TEMP_DIR / sroot.LANG_DICT_TXT.name}

    # convert
    ct2-fairseq-converter \\
        --model_path {best_ckpt_fname} \\
        --data_dir {sroot.CT2_TEMP_DIR} \\
        --fixed_dictionary {sroot.DATA_DICT_TXT} \\
        --output_dir {sroot.CT2_MODEL_DIR} \\
        --force

    # copy spm
    cp {sroot.SPM_MODEL_FILE} {sroot.CT2_MODEL_DIR / sroot.SPM_MODEL_FILE.name}
    cp {sroot.SPM_VOCAB_FILE} {sroot.CT2_MODEL_DIR / sroot.SPM_VOCAB_FILE.name}

    # copy tensorboard
    cp -a {sroot.FAIRSEQ_TENSORBOARD_DIR} {sroot.CT2_MODEL_DIR / sroot.FAIRSEQ_TENSORBOARD_DIR.name}

    # copy best_ckpt
    cp {best_ckpt_fname} {sroot.CT2_MODEL_DIR / best_ckpt_fname.name}

    # check
    tree -h --du -a {sroot.CT2_MODEL_DIR}
    """
    sroot.SCRIPT_DIR.mkdir(exist_ok=True, parents=True)
    utils.write_sh(sroot.SCRIPT_DIR / "convert_fairseq_to_ct2.sh", script)


def run_convert_fairseq_to_ct2() -> None:
    fname = sroot.SCRIPT_DIR / "convert_fairseq_to_ct2.sh"
    assert fname.is_file(), f"{fname} not found"
    utils.subprocess_run(f"bash {fname}")


def main() -> None:
    gen_model_dir()
    if 0:
        utils.open_code(sroot.FAIRSEQ_MODEL_DIR)
    cmd_convert_fairseq_to_ct2()
    run_convert_fairseq_to_ct2()
    utils.log_written(sroot.CT2_MODEL_DIR / "model.bin")  # 231.9M, 5f9373b2
    if 0:
        utils.open_code(sroot.CT2_MODEL_DIR)


if __name__ == "__main__":
    if hasattr(sys, "ps1"):
        pretty.install()
        reload(utils)
        reload(sroot)
    else:
        with logger.catch(onerror=lambda _: sys.exit(1)):
            # python -m src.train.mt_h2ke.ajd_klc.fairseq_convert
            typer.run(main)
