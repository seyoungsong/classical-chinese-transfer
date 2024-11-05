import sys
from importlib import reload

import typer
from loguru import logger
from rich import pretty

import src.crawl.hue_ner.root as sroot
from src import utils


def cmd_gen_dl_dir() -> None:
    # 3GB, 12 files
    local_dir = sroot.CRAWL_DL_DIR
    local_dir.mkdir(parents=True, exist_ok=True)
    cmd = f"""
    rclone sync "g1:temp8/dataset/Named Entity Recognition" {local_dir} \\
        --progress \\
        --transfers 10 \\
        --track-renames \\
        --exclude ".DS_Store" \\
        --dry-run
    """.strip()
    sroot.SCRIPT_DIR.mkdir(parents=True, exist_ok=True)
    utils.write_sh(sroot.SCRIPT_DIR / "download.sh", cmd)


def main() -> None:
    # crawl all data pages (unzip github master zip)
    cmd_gen_dl_dir()
    logger.debug(utils.folder_size(sroot.CRAWL_DL_DIR))  # 474.7M


if __name__ == "__main__":
    if hasattr(sys, "ps1"):
        pretty.install()
        reload(utils)
        reload(sroot)
    else:
        with logger.catch(onerror=lambda _: sys.exit(1)):
            # python -m src.crawl.hue_ner.crawl_dl
            typer.run(main)
