import sys
from importlib import reload
from pathlib import Path

import sentencepiece
import typer
from loguru import logger
from rich import pretty

from src import utils


def load_spm_model(f: Path) -> sentencepiece.SentencePieceProcessor:
    sp = sentencepiece.SentencePieceProcessor(str(f.resolve()))
    return sp


def spm_encode(sp: sentencepiece.SentencePieceProcessor, s: str) -> str:
    if not isinstance(s, str):
        return s
    s = " ".join(sp.EncodeAsPieces(s))
    return s


def spm_decode(sp: sentencepiece.SentencePieceProcessor, s: str) -> str:
    if not isinstance(s, str):
        return s
    s = sp.DecodePieces(s.strip().split(" "))
    return s


def main() -> None:
    pass


if __name__ == "__main__":
    if hasattr(sys, "ps1"):
        pretty.install()
        reload(utils)
    else:
        with logger.catch(onerror=lambda _: sys.exit(1)):
            typer.run(main)
