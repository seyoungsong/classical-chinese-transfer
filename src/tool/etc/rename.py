import shutil
import sys
from importlib import reload

import typer
from loguru import logger
from rich import pretty

from src import utils


def rename_result_files() -> None:
    src_list = sorted(list(utils.RESULT_ROOT_DIR.rglob("f1_reduce4.json")))
    tgt_list = [p.parent / "f1.json" for p in src_list]
    pairs = list(zip(src_list, tgt_list, strict=True))

    # copy
    for f_src, f_tgt in pairs:
        shutil.copy2(f_src, f_tgt)
        utils.log_written(f_tgt)

    # delete
    for f_src, _ in pairs:
        f_src.unlink()
        logger.debug(f"delete: {f_src}")


def main() -> None:
    rename_result_files()


if __name__ == "__main__":
    if hasattr(sys, "ps1"):
        pretty.install()
        reload(utils)
    else:
        with logger.catch(onerror=lambda _: sys.exit(1)):
            typer.run(main)
