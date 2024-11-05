import shutil
import sys
from importlib import reload

import humanize
import typer
from loguru import logger
from rich import pretty

import src.crawl.niu_cc_zh.root as sroot
from src import utils


def cmd_gen_dl_dir() -> None:
    git_url = "https://github.com/NiuTrans/Classical-Modern"
    git_dir = sroot.CRAWL_DL_DIR
    git_dir.parent.mkdir(exist_ok=True, parents=True)
    cmd = f"""
    git clone --depth=1 {git_url} {git_dir}
    """
    fname = sroot.SCRIPT_DIR / "gen_CRAWL_DL_DIR.sh"
    fname.parent.mkdir(exist_ok=True, parents=True)
    utils.write_sh(fname, cmd)


def cmd_gen_7z() -> None:
    sroot.CRAWL_7Z.parent.mkdir(exist_ok=True, parents=True)
    free_space = humanize.naturalsize(
        shutil.disk_usage(sroot.CRAWL_7Z.parent).free, gnu=True
    )
    cmd = f"""
    df -h {sroot.CRAWL_7Z.parent} # {free_space}
    du -hd0 {sroot.CRAWL_DL_DIR}
    7za a {sroot.CRAWL_7Z} {sroot.CRAWL_DL_DIR}
    """
    fname = sroot.SCRIPT_DIR / "gen_CRAWL_7Z.sh"
    fname.parent.mkdir(exist_ok=True, parents=True)
    utils.write_sh(fname, cmd)


def run_gen_7z() -> None:
    fname = sroot.SCRIPT_DIR / "gen_CRAWL_7Z.sh"
    logger.info(f"Run: {fname}")
    utils.subprocess_run(f"bash {fname}")


def main() -> None:
    # crawl all data pages (git clone)
    if 0:
        cmd_gen_dl_dir()  # eta 4h
        logger.debug(utils.folder_size(sroot.CRAWL_DL_DIR))  # 864.9M
        cmd_gen_7z()
        run_gen_7z()
        utils.log_written(sroot.CRAWL_7Z)  # 414.0M, 79a88afc
    try:
        # tmux new -s crawl
        # conda run --no-capture-output -n mmm python -m src.crawl.niu_cc_zh.crawl_dl
        # tmux attach-session -t crawl
        cmd_gen_dl_dir()
    finally:
        utils.notify(title="crawl", body=f"Done: {utils.datetime_kst().isoformat()}")


if __name__ == "__main__":
    if hasattr(sys, "ps1"):
        pretty.install()
        reload(utils)
        reload(sroot)
    else:
        with logger.catch(onerror=lambda _: sys.exit(1)):
            # python -m src.crawl.niu_cc_zh.crawl_dl
            typer.run(main)
