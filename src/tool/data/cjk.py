import json
import re
import unicodedata
from pathlib import Path

_CTEXT_JSON = Path("./src/tool/data/ctext.json").resolve()


_pairs: list[list[str]] = json.loads(_CTEXT_JSON.read_text(encoding="utf-8"))
assert len(_pairs) == 395, "len _chinese_pairs not 395"
_dict = {k: v for k, v in _pairs}
_CHINESE_TRANS = str.maketrans(_dict)


def _repl_nfkc(match: re.Match[str]) -> str:
    return unicodedata.normalize("NFKC", match.group(0))


def _normalize_chinese(s: str) -> str:
    # Normalize chinese characters
    # Conversion data listed on https://ctext.org/faq/normalization
    # https://github.com/daangn/normalize-cjk/blob/main/src/chinese.ts
    return s.translate(_CHINESE_TRANS)


def _normalize_japanese(s: str) -> str:
    # Normalize Katakana characters.
    # Replace all half-width kana with full-width forms.
    # https://github.com/daangn/normalize-cjk/blob/main/src/japanese.ts
    katakana_pattern = r"[\uff60-\uff9f]+"
    return re.sub(katakana_pattern, _repl_nfkc, s)


def _normalize_korean(s: str) -> str:
    # Normalize Hangul (Hangeul, Korean alphabet) characters to NFKC(Normalization Form Compatibility Composition).
    # https://github.com/daangn/normalize-cjk/blob/main/src/korean.ts
    hangul_pattern = (
        r"[\u1100-\u11ff\u3130-\u318f\u3200-\u321e\u3260-\u327f\uffa0-\uffdc\uffe6]+"
    )
    return re.sub(hangul_pattern, _repl_nfkc, s)


def normalize_cjk(s: str) -> str:
    # https://github.com/daangn/normalize-cjk/blob/main/src/all.ts
    return _normalize_chinese(_normalize_japanese(_normalize_korean(s)))
