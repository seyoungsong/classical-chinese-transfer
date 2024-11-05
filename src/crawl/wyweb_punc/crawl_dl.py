import shutil
import sys
from importlib import reload
from pathlib import Path

import humanize
import typer
from loguru import logger
from rich import pretty

import src.crawl.wyweb_punc.root as sroot
from src import utils


def cmd_gen_dl_dir() -> None:
    # find download file
    # https://github.com/baudzhou/WYWEB/archive/fe14b54f3959877dcbdde016c3e9c7ae8642cf10.zip
    user_dir = Path.home() / "Downloads"
    assert user_dir.is_dir()
    fnames = sorted(list(user_dir.rglob("WYWEB*.zip")))
    fname = fnames[0]
    if 0:
        utils.log_written(fname)  # 106.2M, 52c2e3db

    sroot.CRAWL_DL_DIR.mkdir(exist_ok=True, parents=True)
    free_space = humanize.naturalsize(
        shutil.disk_usage(sroot.CRAWL_DL_DIR).free, gnu=True
    )
    cmd = f"""
    # FREE: {free_space}
    7za l {fname}
    7za x {fname} -o{sroot.CRAWL_DL_DIR} -y
    """
    sroot.SCRIPT_DIR.mkdir(exist_ok=True, parents=True)
    utils.write_sh(sroot.SCRIPT_DIR / "unzip_github_zip.sh", cmd)


def main() -> None:
    # crawl all data pages (unzip github master zip)
    cmd_gen_dl_dir()
    logger.debug(utils.folder_size(sroot.CRAWL_DL_DIR))  # 224.6M


if __name__ == "__main__":
    if hasattr(sys, "ps1"):
        pretty.install()
        reload(utils)
        reload(sroot)
    else:
        with logger.catch(onerror=lambda _: sys.exit(1)):
            # python -m src.crawl.wyweb_punc.crawl_dl
            typer.run(main)
