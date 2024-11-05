import sys
from importlib import reload

import pandas as pd
import typer
from loguru import logger
from rich import pretty

import src.eval.corpus.root as sroot
import src.tool.data as dtool
import src.tool.data.cjk as dcjk
import src.tool.eval as etool
from src import utils


def gen_stat() -> None:
    # fnames
    dir1 = sroot.RESULT_DIR
    fnames = sorted(dir1.rglob("char_category.json"))
    fnames = [p for p in fnames if "agg" not in p.parent.name]

    # read
    df_list = [utils.read_df(fname) for fname in fnames]
    df = pd.concat(df_list, ignore_index=True)
    df.sample(1).iloc[0].to_dict()

    # get text
    text = "".join(df["chars"])
    text = "".join(sorted(set(text)))

    # get char
    etool.report_char_punc_freq(texts=[text], output_dir=sroot.RESULT_DIR / "agg")

    # basic normal
    text1 = utils.squeeze_whites(text)
    text1 = dcjk.normalize_cjk(text1)
    text1 = "".join(sorted(set(text1)))
    etool.report_char_punc_freq(texts=[text1], output_dir=sroot.RESULT_DIR / "agg_v1")

    # full normal
    text1 = dtool.normalize_str(text)
    text1 = "".join(sorted(set(text1)))
    etool.report_char_punc_freq(texts=[text1], output_dir=sroot.RESULT_DIR / "agg_v2")


def main() -> None:
    gen_stat()


if __name__ == "__main__":
    if hasattr(sys, "ps1"):
        pretty.install()
        reload(utils)
        reload(sroot)
        reload(dtool)
    else:
        with logger.catch(onerror=lambda _: sys.exit(1)):
            # python -m src.eval.corpus.table
            typer.run(main)
