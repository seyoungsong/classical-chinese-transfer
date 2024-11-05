from pathlib import Path

_DATA_DIR = Path("./src/tool/punc").resolve()


PUNC_NAMES_JSON = _DATA_DIR / "punc_names.json"
PUNC_PRECEDENCE_JSON = _DATA_DIR / "punc_precedence.json"
PUNC_REDUCTION_JSON = _DATA_DIR / "punc_reduction.json"
PUNC_CORPUS_JSON = _DATA_DIR / "punc_selection.json"


NOT_PUNC = "【】〔〕#&@※"
