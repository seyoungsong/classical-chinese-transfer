import re
import sys
from importlib import reload
from pathlib import Path

import typer
from loguru import logger
from rich import pretty

import src.train.mt_h2ke.ajd_klc.root as sroot
from src import utils

# arch
# transformer_wmt_en_de: 65105920 (65.1 million)
# transformer_wmt_en_de_big: 218292224 (218.3 million)
MODEL_ARCH = "transformer_wmt_en_de"
SEED = 42
TENSORBOARD_PORT = 12343


# update
# gpu:6, step:400k -> 1epoch = 2500step = 40min -> 160epoch = 400k step = 106h = 4.4d
# rtx2080ti:2, 1epoch=2526step = 25min -> 150epoch=25*150=62h=2.6d
# result: 161epoch = 421133sec = 4.9day
MAX_EPOCH = 150  # MAX_UPDATE = 400000
LEARNING_RATE = 0.0005
STOP_MIN_LR = 1e-09
WARMUP_INIT_LR = 1e-07
WARMUP_UPDATES = 10000


# save
SAVE_INTERVAL = 1
KEEP_LAST_EPOCHS = 2
KEEP_BEST_CHECKPOINTS = 2


# batch size
# base: 3000 good?, big: 1100 OOM
MAX_TOKENS = 3584  # MAX_TOKENS = 3000
MAX_POSITIONS = sroot.MAX_LEN


# gpu
GPU_DEVICES = [0, 1, 2, 3]
GPU_DEVICE_ONE = GPU_DEVICES[0]
# simulate training on 1x24=24 GPUs
UPDATE_FREQ_TARGET = 16
# simulate training on 6x4=24 GPUs
UPDATE_FREQ_TORCHRUN = UPDATE_FREQ_TARGET // len(GPU_DEVICES)


def cmd_fairseq_train() -> None:
    # https://github.com/facebookresearch/fairseq/blob/main/examples/translation/README.md#multilingual-translation
    # https://github.com/facebookresearch/fairseq/blob/main/examples/multilingual/finetune_multilingual_model.sh
    # https://github.com/facebookresearch/fairseq/blob/main/examples/multilingual/README.md#training

    script_1gpu = f"""
    CUDA_VISIBLE_DEVICES={GPU_DEVICE_ONE} fairseq-train \\
        {sroot.FAIRSEQ_BIN_DIR} \\
        --adam-betas '(0.9, 0.98)' \\
        --adam-eps 1e-06 \\
        --arch {MODEL_ARCH} \\
        --attention-dropout 0.1 \\
        --criterion label_smoothed_cross_entropy \\
        --ddp-backend legacy_ddp \\
        --decoder-langtok \\
        --decoder-normalize-before \\
        --dropout 0.3 \\
        --encoder-langtok src \\
        --encoder-normalize-before \\
        --keep-best-checkpoints {KEEP_BEST_CHECKPOINTS} \\
        --keep-last-epochs {KEEP_LAST_EPOCHS} \\
        --label-smoothing 0.2 \\
        --lang-dict {sroot.LANG_DICT_TXT} \\
        --lang-pairs {sroot.LANG_PAIRS_TXT} \\
        --layernorm-embedding \\
        --log-interval 1 \\
        --lr {LEARNING_RATE} \\
        --lr-scheduler inverse_sqrt \\
        --max-epoch {MAX_EPOCH} \\
        --max-source-positions {MAX_POSITIONS} \\
        --max-target-positions {MAX_POSITIONS} \\
        --max-tokens {MAX_TOKENS} \\
        --memory-efficient-fp16 \\
        --optimizer adam \\
        --sampling-method temperature \\
        --sampling-temperature 1.5 \\
        --save-dir {sroot.FAIRSEQ_CKPT_DIR} \\
        --save-interval {SAVE_INTERVAL} \\
        --seed {SEED} \\
        --share-all-embeddings \\
        --share-decoder-input-output-embed \\
        --skip-invalid-size-inputs-valid-test \\
        --stop-min-lr {STOP_MIN_LR} \\
        --task translation_multi_simple_epoch \\
        --tensorboard-logdir {sroot.FAIRSEQ_TENSORBOARD_DIR} \\
        --update-freq {UPDATE_FREQ_TARGET} \\
        --warmup-init-lr {WARMUP_INIT_LR} \\
        --warmup-updates {WARMUP_UPDATES} \\
        --weight-decay 0.0001
    """
    # mkdir
    sroot.FAIRSEQ_CKPT_DIR.mkdir(exist_ok=True, parents=True)
    sroot.FAIRSEQ_TENSORBOARD_DIR.mkdir(exist_ok=True, parents=True)

    # write script
    sroot.SCRIPT_DIR.mkdir(exist_ok=True, parents=True)
    clear_script_1gpu = "clear && " + script_1gpu.strip()
    utils.write_sh(sroot.SCRIPT_DIR / "fairseq_train_1gpu.sh", clear_script_1gpu)

    # distributed training
    CUDA_VISIBLE_DEVICES = ",".join([str(i) for i in GPU_DEVICES])
    snippet = f"""
    OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES={CUDA_VISIBLE_DEVICES} torchrun --nproc_per_node {len(GPU_DEVICES)} $(which fairseq-train)
    """.strip()
    script_torchrun = re.sub(r"^.*fairseq-train", snippet, script_1gpu.strip())
    script_torchrun = re.sub(
        r"--update-freq \d+", f"--update-freq {UPDATE_FREQ_TORCHRUN}", script_torchrun
    )

    script_torchrun_line = utils.shfmt_oneline(script_torchrun)

    tname = f"fairseq_train_{sroot.MODEL_DIR.name}"

    # tmux
    script_tmux = f"""
    # check
    tmux ls

    # run
    tmux new-session -d -s {tname}
    tmux send-keys -t {tname} "cd {Path.cwd().resolve()} && conda run --no-capture-output -n mmm {script_torchrun_line} && conda run --no-capture-output -n mmm apprise -vv -t 'Done' -b '{tname}' '{utils.DISCORD_URL}'" C-m

    # check
    tmux attach-session -t {tname}
    # tmux kill-session -t {tname} || true

    # check
    tree {sroot.FAIRSEQ_CKPT_DIR}
    """
    utils.write_sh(sroot.SCRIPT_DIR / "fairseq_train_torchrun.sh", script_tmux)


def cmd_tensorboard() -> None:
    ip_addr = utils.get_ip_addr()
    script = f"""
    tensorboard \\
        --logdir {sroot.FAIRSEQ_TENSORBOARD_DIR} \\
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
    cmd_fairseq_train()
    cmd_tensorboard()


if __name__ == "__main__":
    if hasattr(sys, "ps1"):
        pretty.install()
        reload(utils)
        reload(sroot)
    else:
        with logger.catch(onerror=lambda _: sys.exit(1)):
            # python -m src.train.mt_h2ke.ajd_klc.fairseq_train
            typer.run(main)
