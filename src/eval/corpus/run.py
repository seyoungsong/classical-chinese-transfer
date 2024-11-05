import sys
from importlib import reload
from pathlib import Path

import typer
from loguru import logger
from rich import pretty

import src.eval.corpus.root as sroot
from src import utils


def cmd_all_stat() -> None:
    names_str = "ajd, drs, drri, klc"
    py_names = [s.strip() for s in names_str.split(",")]
    tmux_names = [f"stat_{s}" for s in py_names]
    cmds = [
        f"""
        tmux new-session -d -s {t}
        tmux send-keys -t {t} "cd {Path.cwd().resolve()} && conda run --no-capture-output -n mmm python -m src.eval.corpus.{s}; if [ \\$? -eq 0 ]; then tmux kill-session -t {t}; fi" C-m
        # tmux attach-session -t {t}
        # tmux kill-session -t {t} || true
        """.strip()
        for t, s in zip(tmux_names, py_names, strict=True)
    ]
    cmds.append("# tmux kill-server")
    cmd = "\n\n".join(cmds)

    fname = sroot.SCRIPT_DIR / "all_stat.sh"
    fname.parent.mkdir(parents=True, exist_ok=True)
    utils.write_sh(fname, cmd)


def run_all_stat() -> None:
    fname = sroot.SCRIPT_DIR / "all_stat.sh"
    cmd = f"bash {fname}"
    logger.debug(cmd)
    utils.subprocess_run(cmd)


def main() -> None:
    cmd_all_stat()
    run_all_stat()


if __name__ == "__main__":
    if hasattr(sys, "ps1"):
        pretty.install()
        reload(utils)
        reload(sroot)
    else:
        with logger.catch(onerror=lambda _: sys.exit(1)):
            # python -m src.eval.corpus.run
            typer.run(main)
