import json
import sys
import unicodedata
from importlib import reload
from pathlib import Path

import pandas as pd
import typer
from loguru import logger
from rich import pretty

from src import utils

_UNICODE_JSON = Path("./src/tool/data/unicode.json").resolve()
_UNICODE_TSV = Path("./src/tool/data/unicode.tsv").resolve()
_dicts: list[dict[str, str]] = json.loads(_UNICODE_JSON.read_text(encoding="utf-8"))
_dict = {d["src"]: d["tgt"] for d in _dicts}
_UNICODE_TRANS = str.maketrans(_dict)


def _unicode_name(c: str) -> str:
    if len(c) == 0:
        return ""
    return unicodedata.name(c).title().replace(" ", "")


def _normalize_punc(s: str) -> str:
    # Normalize punctuation characters only (NFKC)
    chars = []
    for c in s:
        cat = unicodedata.category(c)
        if cat.startswith("P"):
            c = unicodedata.normalize("NFKC", c)
        chars.append(c)
    return "".join(chars)


def normalize_unicode(s: str) -> str:
    # Normalize punctuation and etc.
    s1 = _normalize_punc(s)
    s2 = s1.translate(_UNICODE_TRANS)
    return s2


def gen_unicode_json() -> None:
    # original
    src2tgt = {
        "ᆞ∙⋅・ㆍ･": "·",
        "‐–—―−─－": "-",
        "‘’": "'",
        "“”‟": '"',
        "․．": ".",
        "∶：": ":",
        "！": "!",
        "（": "(",
        "）": ")",
        "，": ",",
        "／": "/",
        "；": ";",
        "？": "?",
        "［": "[",
        "］": "]",
        "｡": "。",
        "｢": "「",
        "｣": "」",
    }
    src2tgt = {"".join(sorted(set(k) - {v})): v for k, v in src2tgt.items()}
    src2tgt = utils.sort_dict(src2tgt)

    # flatten
    pairs = list(utils.flatten([[[k1, v] for k1 in k] for k, v in src2tgt.items()]))
    df = pd.DataFrame(pairs, columns=["src", "tgt"])
    df.drop_duplicates(inplace=True)
    df.sort_values(by=["tgt", "src"], inplace=True, ignore_index=True)

    # add name
    df["src_name"] = df["src"].apply(_unicode_name)
    df["tgt_name"] = df["tgt"].apply(_unicode_name)

    # drop if NFKC is enough
    df["nfkc"] = df["src"].apply(_normalize_punc)
    df["nfkc_name"] = df["nfkc"].apply(_unicode_name)
    idx = df["tgt"] == df["nfkc"]
    df = df[~idx].reset_index(drop=True)

    # clear NFKC if no change
    idx = df["src"] == df["nfkc"]
    df.loc[idx, "nfkc"] = ""
    df["nfkc_name"] = df["nfkc"].apply(_unicode_name)

    # drop if NFKC is included in src
    idx = df["nfkc"].isin(df["src"])
    df = df[~idx].reset_index(drop=True)
    df.drop(columns=["nfkc", "nfkc_name"], inplace=True)

    # manual drop
    drop_src_name_list = ["BoxDrawingsLightHorizontal"]
    idx = df["src_name"].isin(drop_src_name_list)
    df = df[~idx].reset_index(drop=True)

    # save
    utils.write_json(_UNICODE_JSON, df.to_dict(orient="records"))
    utils.write_df(_UNICODE_TSV, df)


def main() -> None:
    if 0:
        gen_unicode_json()


if __name__ == "__main__":
    if hasattr(sys, "ps1"):
        pretty.install()
        reload(utils)
    else:
        with logger.catch(onerror=lambda _: sys.exit(1)):
            typer.run(main)
