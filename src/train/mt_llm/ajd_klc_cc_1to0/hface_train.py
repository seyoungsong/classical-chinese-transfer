import platform
import sys
from importlib import reload
from pathlib import Path

import typer
from loguru import logger
from rich import pretty

import src.train.mt_llm.ajd_klc_cc_1to0.root as sroot
from src import utils

IS_A100 = platform.node() == "anonymous"
TENSORBOARD_PORT = 12341
GPU_DEVICE_ONE = 1 if not IS_A100 else 0


def cmd_hface_train() -> None:
    # pass info by file
    hface_info = {
        "DATASET_FILE": str(sroot.DATASET_PQ),
        "HFACE_TRAIN_DIR": str(sroot.HFACE_TRAIN_DIR),
        "HFACE_LORA_DIR": str(sroot.HFACE_LORA_DIR),
    }
    utils.write_json(sroot.HFACE_INFO_JSON, hface_info)

    # data
    dataset_file = sroot.DATASET_PQ
    lora_method = "lora" if IS_A100 else "qlora"
    pcli_file = Path(sroot.__file__).parent / "unsloth_qlora.py"

    # check
    if not dataset_file.is_file():
        logger.warning(f"dataset_file not found: {dataset_file}")
    if not pcli_file.is_file():
        logger.warning(f"pcli_file not found: {pcli_file}")

    # script
    script_1gpu = f"""
    CUDA_VISIBLE_DEVICES={GPU_DEVICE_ONE} conda run --no-capture-output -n unsloth_env python {pcli_file} --method {lora_method}
    """
    # mkdir
    sroot.HFACE_TRAIN_DIR.mkdir(exist_ok=True, parents=True)
    sroot.HFACE_LORA_DIR.mkdir(exist_ok=True, parents=True)

    # write script
    sroot.SCRIPT_DIR.mkdir(exist_ok=True, parents=True)
    clear_script_1gpu = "clear && " + script_1gpu.strip()
    utils.write_sh(sroot.SCRIPT_DIR / "hface_train_1gpu.sh", clear_script_1gpu)

    # distributed training
    script_1gpu_line = utils.shfmt_oneline(script_1gpu)

    tname = f"hface_train_{sroot.MODEL_DIR.name}"

    # tmux
    script_tmux = f"""
    # check
    tmux ls

    # run
    tmux new-session -d -s {tname}
    tmux send-keys -t {tname} "cd {Path.cwd().resolve()} && {script_1gpu_line} && conda run --no-capture-output -n mmm apprise -vv -t 'Done' -b '{tname}' '{utils.DISCORD_URL}'" C-m

    # check
    tmux attach-session -t {tname}
    # tmux kill-session -t {tname} || true

    # check
    tree {sroot.HFACE_TRAIN_DIR}
    tree {sroot.HFACE_LORA_DIR}
    """
    utils.write_sh(sroot.SCRIPT_DIR / "hface_train_1gpu_tmux.sh", script_tmux)


def cmd_tensorboard() -> None:
    ip_addr = utils.get_ip_addr()
    script = f"""
    tensorboard \\
        --logdir {sroot.HFACE_TRAIN_DIR} \\
        --port {TENSORBOARD_PORT} \\
        --bind_all \\
        --load_fast true
    """
    script_line = utils.shfmt_oneline(script)

    # tmux
    tname = f"tensorboard_{sroot.MODEL_DIR.name}"
    script_tmux = f"""
    # check
    # tmux ls
    # tmux kill-session -t {tname} || true

    # run
    tmux new-session -d -s {tname}
    tmux send-keys -t {tname} "cd {Path.cwd().resolve()} && conda run --no-capture-output -n mmm {script_line}" C-m

    # check
    # tmux attach-session -t {tname}
    # http://{ip_addr}:{TENSORBOARD_PORT}
    # http://127.0.0.1:{TENSORBOARD_PORT}
    """
    sroot.SCRIPT_DIR.mkdir(exist_ok=True, parents=True)
    utils.write_sh(sroot.SCRIPT_DIR / "tensorboard.sh", script_tmux)


def main() -> None:
    cmd_hface_train()
    cmd_tensorboard()


if __name__ == "__main__":
    if hasattr(sys, "ps1"):
        pretty.install()
        reload(utils)
        reload(sroot)
    else:
        with logger.catch(onerror=lambda _: sys.exit(1)):
            # python -m src.train.mt_llm.ajd_klc_cc_1to0.hface_train
            typer.run(main)
