import re
import sys
from importlib import reload

import typer
from loguru import logger
from rich import pretty

import src.tool.data.cjk as dcjk
import src.tool.data.unicode as dunicode
import src.tool.eval as etool
from src import utils


def normalize_str(s: str, is_start: bool = True) -> str:
    # 모든 CJK 코퍼스에 적용될 수 있는 전처리. 기존 것 적극 활용.

    # pandas 용 예외처리
    if not isinstance(s, str):
        return s

    # 시작: 공백, 뉴라인 압축
    s = utils.squeeze_whites(s)

    # CJK 정규화 (당근): 狀 > 狀, 不 > 不
    s = dcjk.normalize_cjk(s)

    # punc NFKC 정규화, 유사한 unicode 문자 통합(아래아, dash 등)
    s = dunicode.normalize_unicode(s)

    # etc: 격일자 ○가 문장 앞에 있는 경우 제거
    if is_start:
        s = re.sub(r"^[(〔〈]?○[)〕〉]?\s*", "", s).strip()

    # 마무리
    s = utils.squeeze_whites(s)

    return s


def normalize_xml(s_xml: str) -> str:
    # pandas 용 예외처리
    if not isinstance(s_xml, str):
        return s_xml

    # convert to items
    items = etool.xml2items(s_xml=s_xml)

    # normalize
    texts2 = [normalize_str(s=d["text"], is_start=i == 0) for i, d in enumerate(items)]  # type: ignore

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
