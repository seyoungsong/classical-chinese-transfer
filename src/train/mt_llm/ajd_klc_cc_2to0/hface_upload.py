import sys
from importlib import reload

import typer
from loguru import logger
from rich import pretty

import src.train.mt_llm.ajd_klc_cc_2to0.root as sroot
from src import utils


def cmd_hface_upload() -> None:
    # local
    local_path = sroot.HFACE_MODEL_DIR.resolve()
    local_path.mkdir(parents=True, exist_ok=True)

    # Usage: huggingface-cli upload [repo_id] [local_path] [path_in_repo]
    script = f"""
    # check
    conda run --no-capture-output -n mmm huggingface-cli whoami
    rclone tree {local_path}  # {utils.folder_size(local_path)}

    # upload
    conda run --no-capture-output -n mmm huggingface-cli upload \\
        --repo-type model \\
        --private \\
        {sroot.HFACE_REPO_ID} \\
        {local_path} .

    # Change model visibility
    # https://huggingface.co/{sroot.HFACE_REPO_ID}/settings

    # download
    conda run --no-capture-output -n mmm huggingface-cli download \\
        {sroot.HFACE_REPO_ID} \\
        --local-dir {local_path} \\
        --local-dir-use-symlinks False
    """
    sroot.SCRIPT_DIR.mkdir(parents=True, exist_ok=True)
    utils.write_sh(sroot.SCRIPT_DIR / "hface_upload.sh", script)


def main() -> None:
    cmd_hface_upload()


if __name__ == "__main__":
    if hasattr(sys, "ps1"):
        pretty.install()
        reload(utils)
        reload(sroot)
    else:
        with logger.catch(onerror=lambda _: sys.exit(1)):
            # python -m src.train.mt_llm.ajd_klc_cc_2to0.hface_upload
            typer.run(main)
