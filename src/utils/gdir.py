from pathlib import Path
from typing import Final

SCRIPT_ROOT_DIR: Final = Path("scripts").resolve()
RESULT_ROOT_DIR: Final = Path("results").resolve()

# anonymous-out4/
__OUT_ROOT_DIR_LOCAL: Final = Path("../anonymous-out4").resolve()
__OUT_ROOT_DIR_SSD: Final = Path("/ssd/anonymous-out4").resolve()
__OUT_ROOT_DIR: Final = (
    __OUT_ROOT_DIR_SSD if __OUT_ROOT_DIR_SSD.parent.is_dir() else __OUT_ROOT_DIR_LOCAL
)

# anonymous-out4/dataset/
DATASET_DIR: Final = __OUT_ROOT_DIR / "dataset"
OLD_DATASET_DIR: Final = __OUT_ROOT_DIR / "old_dataset"

# anonymous-out4/model/
MODEL_ROOT_DIR: Final = __OUT_ROOT_DIR / "model"

# anonymous-out4/temp/
TEMP_DIR: Final = __OUT_ROOT_DIR / "temp"
TEMP_TXT: Final = TEMP_DIR / "temp.txt"
TEMP_JSON: Final = TEMP_DIR / "temp.json"
TEMP_JSONL: Final = TEMP_DIR / "temp.jsonl"
TEMP_HTML: Final = TEMP_DIR / "temp.html"
TEMP_XML: Final = TEMP_DIR / "temp.xml"
TEMP_PKL: Final = TEMP_DIR / "temp.pkl.zst"
TEMP_TSV: Final = TEMP_DIR / "temp.tsv"
