import re
import sys
from importlib import reload

import typer
from loguru import logger
from rich import pretty
from transformers import AutoTokenizer, NllbTokenizerFast, PreTrainedTokenizerFast

import src.tool.data.cjk as dcjk
import src.tool.eval as etool
from src import utils


def deprecated_get_fix_char() -> dict[str, str]:
    src2tgt = {
        "ᆞ∙⋅・ㆍ･": "·",
        "‐–—―−─－": "-",
        "‘’": "'",
        "“”‟": '"',
        "․": ".",
        "…": "...",
        "∶：": ":",
        "∼": "~",
        "■□▨▩": "▣",
        "◯〇": "○",
        "\ufeff": "",
        "！": "!",
        "（": "(",
        "）": ")",
        "＊": "*",
        "，": ",",
        "／": "/",
        "；": ";",
        "？": "?",
        "［": "[",
        "］": "]",
        "｢": "「",
        "｣": "」",
    }
    src2tgt = {"".join(sorted(set(k) - {v})): v for k, v in src2tgt.items()}
    src2tgt = utils.sort_dict(src2tgt)

    pairs = list(utils.flatten([[[k1, v] for k1 in k] for k, v in src2tgt.items()]))
    pairs.sort()
    fix_char = {k: v for k, v in pairs if k != v}
    fix_char = utils.sort_dict(fix_char)
    return fix_char


_DEPRECATED_FIX_CHAR = deprecated_get_fix_char()
_DEPRECATED_FIX_CHAR_TRANS = str.maketrans(_DEPRECATED_FIX_CHAR)


class DeprecatedNormalizer:
    def __init__(self) -> None:
        # https://huggingface.co/docs/transformers/multilingual
        # https://huggingface.co/facebook/nllb-200-3.3B
        self.tokenizer: NllbTokenizerFast = AutoTokenizer.from_pretrained(
            "facebook/nllb-200-3.3B"
        )

    def deprecated_normalize_str(self, s: str) -> str:
        return _deprecated_normalize_str(tokenizer=self.tokenizer, s=s)

    def deprecated_normalize_cjk(self, s: str) -> str:
        return dcjk.normalize_cjk(s)

    def deprecated_normalize_safe(self, s: str) -> str:
        try:
            return _deprecated_normalize_safe(s)
        except Exception as e:
            logger.error(f"e: [{repr(e)}], s: [{s}]")
            raise e


def _deprecated_normalize_safe(s: str) -> str:
    if not isinstance(s, str):
        # pandas 용 예외처리
        return s

    # len을 변경하지 않도록 주의
    len_orig = len(s)

    # 시작: 공백, 뉴라인 압축
    c_list = [utils.squeeze_whites(c) for c in s]
    c_list = [" " if len(c) == 0 else c for c in c_list]
    s = "".join(c_list)
    assert len(s) == len_orig, "len mismatch at step1"

    # 유사한 unicode 문자 통합: 아래아, dash 등
    c_list = [c.translate(_DEPRECATED_FIX_CHAR_TRANS) for c in s]
    c_list = [
        c_norm if len(c_norm) == 1 else c_orig
        for c_norm, c_orig in zip(c_list, s, strict=True)
    ]
    s = "".join(c_list)
    assert len(s) == len_orig, "len mismatch at step2"

    # CJK 정규화 (당근): 狀 > 狀, 不 > 不
    s_temp = dcjk.normalize_cjk(s)
    if len(s_temp) == len_orig:
        s = s_temp
    else:
        s = dcjk._normalize_chinese(s)
    assert len(s) == len_orig, "len mismatch at step3"

    # aesthetic: ○가 문장 앞에 있고, 전체에 1개만 있는 경우, 제거
    pattern = re.compile(r"^[(〔]?○[)〕]?\s*")
    if s.count("○") == 1:
        mat = pattern.match(s)
        if mat:
            lead_len = len(mat.group())
            s = " " * lead_len + s[lead_len:]
    assert len(s) == len_orig, "len mismatch at step4"

    # 마무리
    c_list = [utils.squeeze_whites(c) for c in s]
    c_list = [" " if len(c) == 0 else c for c in c_list]
    s = "".join(c_list)
    assert len(s) == len_orig, "len mismatch at step5"

    return s


def _deprecated_normalize_str(tokenizer: PreTrainedTokenizerFast, s: str) -> str:
    # 모든 CJK 코퍼스에 적용될 수 있는 전처리. 기존 것 적극 활용.
    if not isinstance(s, str):
        # pandas 용 예외처리
        return s

    # 시작: 공백, 뉴라인 압축
    s = utils.squeeze_whites(s)

    # 유사한 unicode 문자 통합: 아래아, dash 등
    s = s.translate(_DEPRECATED_FIX_CHAR_TRANS)

    # CJK 정규화 (당근): 狀 > 狀, 不 > 不
    s = dcjk.normalize_cjk(s)

    # NLLB 정규화 (facebook): … > ..., ：>:, ＊>*, －>-, ․>., ufeff>공백, ﬁ>fi, ﬂ>fl, \x9f>공백, \u200b>공백
    s = tokenizer.backend_tokenizer.normalizer.normalize_str(s)

    # aesthetic: ○가 문장 앞에 있고, 전체에 1개만 있는 경우, 제거
    if s.count("○") == 1:
        s = re.sub(r"^[(〔]?○[)〕]?\s*", "", s).strip()

    # 마무리
    s = utils.squeeze_whites(s)

    return s


def deprecated_normalize_xml(s_xml: str, normalizer: DeprecatedNormalizer) -> str:
    # convert to items
    items = etool.xml2items(s_xml=s_xml)

    # normalize
    texts2 = [normalizer.deprecated_normalize_str(s=d["text"]) for d in items]  # type: ignore

    # update items
    items2 = [dict(d, text=s) for d, s in zip(items, texts2, strict=True)]

    # filter out empty text
    items3 = [d for d in items2 if len(d["text"]) >= 1]  # type: ignore

    # convert to xml
    s_xml_result = etool.items2xml(items=items3)
    return s_xml_result


def main() -> None:
    pass


if __name__ == "__main__":
    if hasattr(sys, "ps1"):
        pretty.install()
        reload(utils)
    else:
        with logger.catch(onerror=lambda _: sys.exit(1)):
            typer.run(main)
