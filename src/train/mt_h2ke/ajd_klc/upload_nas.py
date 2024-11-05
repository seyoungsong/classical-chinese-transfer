import sys
from importlib import reload

import typer
from loguru import logger
from rich import pretty

import src.train.mt_h2ke.ajd_klc.root as sroot
from src import utils


def cmd_upload() -> None:
    # local
    local_path = sroot.MODEL_DIR
    local_path.mkdir(parents=True, exist_ok=True)

    # remote
    remote_path = f"/mnt/nas/anonymous/anonymous/model/{sroot.MODEL_DIR.name}"

    script = f"""
    # check
    rclone tree {local_path}  # {utils.folder_size(local_path)}
    rclone tree {remote_path}

    # upload
    rclone sync {local_path} {remote_path} \\
        --progress \\
        --transfers 10 \\
        --track-renames \\
        --exclude ".DS_Store" \\
        --dry-run

    # download
    rclone sync {remote_path} {local_path} \\
        --progress \\
        --transfers 10 \\
        --track-renames \\
        --exclude ".DS_Store" \\
        --dry-run
    """
    sroot.SCRIPT_DIR.mkdir(parents=True, exist_ok=True)
    utils.write_sh(sroot.SCRIPT_DIR / f"upload_nas_{utils.os_name()}.sh", script)

    # tree
    fname2 = sroot.SCRIPT_DIR / f"upload-{utils.os_name()}.txt"
    utils.subprocess_run(f"tree -a --du -h --prune {local_path} > {fname2}")
    utils.log_written(fname2)


def main() -> None:
    cmd_upload()


if __name__ == "__main__":
    if hasattr(sys, "ps1"):
        pretty.install()
        reload(utils)
        reload(sroot)
    else:
        with logger.catch(onerror=lambda _: sys.exit(1)):
            # python -m src.train.mt_h2ke.ajd_klc.upload_nas
            typer.run(main)
