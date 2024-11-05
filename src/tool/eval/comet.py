import subprocess
from typing import Any

import torch
from loguru import logger

from src import utils

__task_name = "tool_comet_v1"
MODEL_DIR = utils.MODEL_ROOT_DIR / __task_name
SCRIPT_DIR = utils.SCRIPT_ROOT_DIR / __task_name
RESULT_DIR = utils.RESULT_ROOT_DIR / __task_name
COMET_CACHE_DIR = utils.MODEL_ROOT_DIR / "comet_cache"


GPU_DEVICES = [3, 5, 6, 7]
BATCH_SIZE = 32
NUM_WORKERS = 8
COMET_MODEL_ID = "Unbabel/wmt22-comet-da"


def compute_COMET22(
    src1: list[str], hypo: list[str], ref1: list[str]
) -> dict[str, Any]:
    # check
    len_list = [len(src1), len(hypo), len(ref1)]
    assert len(set(len_list)) == 1, f"len not match: {len_list}"

    # config
    CUDA_VISIBLE_DEVICES = ",".join([str(i) for i in GPU_DEVICES])
    if torch.cuda.is_available():
        NUM_GPUS = len(GPU_DEVICES)
    else:
        NUM_GPUS = 0  # cpu-only

    # fname
    src1_txt = MODEL_DIR / "src1.txt"
    hypo_txt = MODEL_DIR / "hypo.txt"
    ref1_txt = MODEL_DIR / "ref1.txt"
    output_json = MODEL_DIR / "output.json"
    script_fname = SCRIPT_DIR / "run_comet_score.sh"

    # script
    # https://unbabel.github.io/COMET/html/running.html
    # https://github.com/Unbabel/COMET
    cmd = f"""
    CUDA_VISIBLE_DEVICES={CUDA_VISIBLE_DEVICES} conda run --no-capture-output -n comet comet-score \\
        --batch_size {BATCH_SIZE} \\
        --gpus {NUM_GPUS} \\
        --model {COMET_MODEL_ID} \\
        --model_storage_path {COMET_CACHE_DIR} \\
        --num_workers {NUM_WORKERS} \\
        --only_system \\
        --references {ref1_txt} \\
        --sources {src1_txt} \\
        --to_json {output_json} \\
        --translations {hypo_txt}
    """.strip()
    script_fname.parent.mkdir(exist_ok=True, parents=True)
    utils.write_sh(script_fname, cmd)

    # prepare
    COMET_CACHE_DIR.mkdir(exist_ok=True, parents=True)
    src1_txt.parent.mkdir(exist_ok=True, parents=True)
    utils.write_str(src1_txt, "\n".join(src1))
    utils.write_str(hypo_txt, "\n".join(hypo))
    utils.write_str(ref1_txt, "\n".join(ref1))

    # compute
    logger.debug(f"run: {script_fname}")
    subprocess.run(f"bash {script_fname}", shell=True)

    # read
    output_dict: dict[str, Any] = utils.read_json(output_json)
    assert len(output_dict.keys()) == 1, "output dict not one key"
    output_list: list[dict[str, Any]] = list(output_dict.values())[0]
    score_list = [d["COMET"] for d in output_list]
    score_raw = sum(score_list) / len(score_list)
    score = round(score_raw * 100, 2)

    # output
    result = {
        "COMET_score": score,
        "COMET_score_raw": score_raw,
        "num_sample": len(hypo),
        "src1_len": sum(map(len, src1)),
        "hypo_len": sum(map(len, hypo)),
        "ref1_len": sum(map(len, ref1)),
    }

    return result
