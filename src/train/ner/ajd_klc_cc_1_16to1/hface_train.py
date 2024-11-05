import os
import re
import sys
from importlib import reload
from pathlib import Path

import typer
from loguru import logger
from rich import pretty

import src.tool.hface.run_ner as pcli
import src.train.ner.ajd_klc_cc_1_16to1.root as sroot
from src import utils

# arch
MODEL_NAME = "SIKU-BERT/sikuroberta"
TENSORBOARD_PORT = 12344

# gpu
GPU_DEVICES = [4]
GPU_DEVICE_ONE = GPU_DEVICES[0]

# update
NUM_TRAIN_EPOCHS = 5
LEARNING_RATE = 2e-5
WARMUP_RATIO = 0.1

# save
SAVE_STEPS = 1 / 50
EVAL_STEPS = SAVE_STEPS
SAVE_TOTAL_LIMIT = 2
LOGGING_STEPS = 1

# vram
PER_DEVICE_TRAIN_BATCH_SIZE = 8
PER_DEVICE_EVAL_BATCH_SIZE = 64
EVAL_ACCUMULATION_STEPS = 16
MAX_SEQ_LENGTH = 512

# logical batch size
TARGET_BATCH_SIZE = 32

# simulate training
GRADIENT_ACCUMULATION_STEPS_1GPU = TARGET_BATCH_SIZE // PER_DEVICE_TRAIN_BATCH_SIZE
GRADIENT_ACCUMULATION_STEPS_4GPU = GRADIENT_ACCUMULATION_STEPS_1GPU // len(GPU_DEVICES)

# etc
SEED = 42
WEIGHT_DECAY = 0.01
DATALOADER_NUM_WORKERS = 4
PREPROCESSING_NUM_WORKERS = round(os.cpu_count() * 0.9)  # type: ignore


def cmd_hface_train() -> None:
    # data
    train_file = sroot.HFACE_JSONL_DIR / "train.json"
    validation_file = sroot.HFACE_JSONL_DIR / "validation.json"

    # check
    if not train_file.is_file():
        logger.warning(f"train_file not found: {train_file}")
    if not validation_file.is_file():
        logger.warning(f"validation_file not found: {validation_file}")

    # ignore test
    if 0:
        test_file = sroot.HFACE_JSONL_DIR / "test.json"
        if not test_file.is_file():
            logger.warning(f"test_file not found: {test_file}")

    # script
    script_1gpu = f"""
    CUDA_VISIBLE_DEVICES={GPU_DEVICE_ONE} python {pcli.__file__} \\
        --data_seed {SEED} \\
        --dataloader_num_workers {DATALOADER_NUM_WORKERS} \\
        --do_eval \\
        --do_train \\
        --eval_accumulation_steps {EVAL_ACCUMULATION_STEPS} \\
        --eval_steps {EVAL_STEPS} \\
        --evaluation_strategy steps \\
        --fp16 \\
        --fp16_full_eval \\
        --gradient_accumulation_steps {GRADIENT_ACCUMULATION_STEPS_1GPU} \\
        --label_column_name ner_tags \\
        --learning_rate {LEARNING_RATE} \\
        --load_best_model_at_end \\
        --logging_steps {LOGGING_STEPS} \\
        --logging_strategy steps \\
        --max_seq_length {MAX_SEQ_LENGTH} \\
        --model_name_or_path {MODEL_NAME} \\
        --num_train_epochs {NUM_TRAIN_EPOCHS} \\
        --output_dir {sroot.HFACE_TRAIN_DIR} \\
        --overwrite_cache \\
        --overwrite_output_dir \\
        --pad_to_max_length \\
        --per_device_eval_batch_size {PER_DEVICE_EVAL_BATCH_SIZE} \\
        --per_device_train_batch_size {PER_DEVICE_TRAIN_BATCH_SIZE} \\
        --preprocessing_num_workers {PREPROCESSING_NUM_WORKERS} \\
        --return_entity_level_metrics \\
        --save_safetensors \\
        --save_steps {SAVE_STEPS} \\
        --save_strategy steps \\
        --save_total_limit {SAVE_TOTAL_LIMIT} \\
        --seed {SEED} \\
        --task_name ner \\
        --text_column_name tokens \\
        --train_file {train_file} \\
        --validation_file {validation_file} \\
        --warmup_ratio {WARMUP_RATIO} \\
        --weight_decay {WEIGHT_DECAY}
    """
    # mkdir
    sroot.HFACE_TRAIN_DIR.mkdir(exist_ok=True, parents=True)

    # write script
    sroot.SCRIPT_DIR.mkdir(exist_ok=True, parents=True)
    clear_script_1gpu = "clear && " + script_1gpu.strip()
    utils.write_sh(sroot.SCRIPT_DIR / "hface_train_1gpu.sh", clear_script_1gpu)

    # distributed training
    CUDA_VISIBLE_DEVICES = ",".join([str(i) for i in GPU_DEVICES])
    snippet = f"""
    CUDA_VISIBLE_DEVICES={CUDA_VISIBLE_DEVICES} python
    """.strip()
    script_4gpu = re.sub(r"^.*python", snippet, script_1gpu.strip())
    script_4gpu = re.sub(
        r"--gradient_accumulation_steps \d+",
        f"--gradient_accumulation_steps {GRADIENT_ACCUMULATION_STEPS_4GPU}",
        script_4gpu,
    )
    # write script
    utils.write_sh(sroot.SCRIPT_DIR / "hface_train_4gpu.sh", script_4gpu)

    script_torchrun_line = utils.shfmt_oneline(script_4gpu)

    tname = f"hface_train_{sroot.MODEL_DIR.name}"

    # tmux
    script_tmux = f"""
    # check
    # tmux ls

    # run
    tmux new-session -d -s {tname}
    tmux send-keys -t {tname} "cd {Path.cwd().resolve()} && conda run --no-capture-output -n mmm {script_torchrun_line} && conda run --no-capture-output -n mmm apprise -vv -t 'Done' -b '{tname}' '{utils.DISCORD_URL}'" C-m

    # check
    # tmux attach-session -t {tname}
    # tmux kill-session -t {tname} || true

    # check
    # tree {sroot.HFACE_TRAIN_DIR}
    """
    utils.write_sh(sroot.SCRIPT_DIR / "hface_train_4gpu_tmux.sh", script_tmux)


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


def main(run: bool = False) -> None:
    cmd_hface_train()
    cmd_tensorboard()
    if run:
        utils.subprocess_run(f'bash {sroot.SCRIPT_DIR / "tensorboard.sh"}')
        utils.subprocess_run(f'bash {sroot.SCRIPT_DIR / "hface_train_4gpu_tmux.sh"}')


if __name__ == "__main__":
    if hasattr(sys, "ps1"):
        pretty.install()
        reload(utils)
        reload(sroot)
    else:
        with logger.catch(onerror=lambda _: sys.exit(1)):
            # python -m src.train.ner.ajd_klc_cc_2to1.hface_train --run
            typer.run(main)
