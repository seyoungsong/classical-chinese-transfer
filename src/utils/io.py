import json
import math
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import humanize
import numpy as np
from loguru import logger

from . import gdir, pure


class _JSONEncoder(json.JSONEncoder):
    def default(self, obj: Any) -> Any:
        if isinstance(obj, Path):
            return str(obj.resolve())
        elif math.isnan(obj) or np.isnan(obj):
            return super().default(None)
        elif isinstance(obj, np.generic):
            return obj.item()
        return super().default(obj)


def temp_diff(s1: str, s2: str) -> None:
    gdir.TEMP_DIR.mkdir(parents=True, exist_ok=True)
    f1 = gdir.TEMP_DIR / "temp_diff_1.txt"
    f2 = gdir.TEMP_DIR / "temp_diff_2.txt"
    f1.write_text(s1, encoding="utf-8")
    f2.write_text(s2, encoding="utf-8")
    pure.code_diff(f1, f2)


def file_size(fname: Path) -> str:
    assert fname.is_file()
    size = humanize.naturalsize(fname.stat().st_size, gnu=True)
    return size


def file_size_raw(fname: Path | str) -> int:
    fname = Path(fname)
    assert fname.is_file()
    size = fname.stat().st_size
    return size


def folder_size(folder: Path) -> str:
    assert folder.is_dir()
    size_sum = sum(p.stat().st_size for p in Path(folder).rglob("*") if p.is_file())
    return humanize.naturalsize(size_sum, gnu=True)


def log_reading(fname: Path) -> None:
    assert fname.is_file()
    size = file_size(fname)
    logger.debug(f"Reading {size} from {fname.resolve()}")


def write_str(fname: Path, s: str) -> None:
    fname.write_text(s, encoding="utf-8")
    log_written(fname)


def read_str(fname: Path) -> str:
    log_reading(fname)
    s = fname.read_text(encoding="utf-8")
    return s


def checksum(fname: Path) -> str:
    assert fname.is_file()
    tool = shutil.which("sha256sum")
    assert tool is not None, "sha256sum not found"
    cmd = f"{tool} {fname}".strip().split()
    result = subprocess.run(cmd, capture_output=True, text=True)
    output = result.stdout
    shasum = output.strip().split()[0]
    assert len(shasum) == 64
    shasum8 = shasum[:8]
    return shasum8


def get_stat(fname: str) -> str:
    p = Path(fname)
    size = file_size(p)
    csum = checksum(p)
    stat = f"{csum} | {size}"
    return stat


def write_json(fname: Path, obj: Any) -> None:
    s = json.dumps(obj, ensure_ascii=False, indent=2, cls=_JSONEncoder)
    fname.write_text(s, encoding="utf-8")
    log_written(fname)


def write_json2(fname: Path, obj: Any) -> None:
    s = json.dumps(obj, ensure_ascii=False, indent=2, cls=_JSONEncoder)
    fname.write_text(s, encoding="utf-8")


def read_json(fname: Path) -> Any:
    log_reading(fname)
    s = fname.read_text(encoding="utf-8")
    d: Any = json.loads(s)
    return d


def read_json2(fname: Path) -> Any:
    s = fname.read_text(encoding="utf-8")
    d: Any = json.loads(s)
    return d


def log_written(fname: Path, etc: str = "") -> None:
    assert fname.is_file()
    size = file_size(fname)
    csum = checksum(fname)
    if len(etc) > 0:
        etc = f", {etc}"
    logger.debug(f"Written {size} to {fname.resolve()} | {size}, {csum}{etc}")


def log_written2(fname: Path) -> None:
    assert fname.is_dir()
    size = folder_size(fname)
    logger.debug(f"Written {size} to {fname.resolve()} | {size}, FOLDER")


def reset_dir(dir_name: Path) -> Path:
    p = Path(dir_name).resolve()
    if p.is_dir():
        logger.warning(f"rm -r {p}")
        shutil.rmtree(p)
    logger.info(f"mkdir -p {p}")
    p.mkdir(parents=True, exist_ok=True)
    return p


def write_sh(fname: Path, script: str) -> None:
    assert fname.suffix == ".sh"
    write_str(fname, script)
    shfmt_file(fname)


def shfmt_str(script: str) -> str:
    with tempfile.NamedTemporaryFile(mode="w+", suffix=".sh") as f:
        p = Path(f.name).resolve()
        p.write_text(script, encoding="utf-8")
        shfmt_file(p)
        script_formatted = p.read_text(encoding="utf-8")
    return script_formatted


def shfmt_oneline(script: str) -> str:
    s1 = shfmt_str(script)
    s2 = s1.replace("\\\n", " ")
    s3 = shfmt_str(s2).strip()
    return s3


def shfmt_file(fname: Path) -> None:
    if shutil.which("shfmt") is None:
        raise FileNotFoundError("shfmt not found")
    assert fname.suffix.lower() == ".sh"
    cmd_str = f"shfmt --write {fname.resolve()}"
    subprocess.run(cmd_str, shell=True)


def get_spm_path(spm_model_dir: Path, prefix: str = "spm") -> tuple[Path, Path]:
    """Get path to sentencepiece model and vocab."""
    p = Path(spm_model_dir).resolve()
    spm_model = p / f"{prefix}.model"
    spm_vocab = p / f"{prefix}.vocab"
    return spm_model, spm_vocab
