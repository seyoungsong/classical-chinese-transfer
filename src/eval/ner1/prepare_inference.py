import sys
from importlib import reload

import typer
from loguru import logger
from pandarallel import pandarallel
from rich import pretty
from tqdm import tqdm

import src.dataset.ner.root
import src.eval.ner1.root as sroot
import src.tool.eval as etool
from src import utils


def gen_dataset_file() -> None:
    # read
    df0 = utils.read_df(src.dataset.ner.root.EVAL_PQ)
    df = df0.copy()
    df.sample(1).iloc[0].to_dict()
    df.columns
    df.info()

    # add input
    df["text"] = df["text_xml"].progress_apply(lambda x: etool.xml2plaintext(x))

    # add id
    assert df["key2"].is_unique
    df.sort_values("key2", inplace=True, ignore_index=True)
    #
    df["id"] = df.index + 1
    digit = len(str(max(df["id"])))
    df["id"] = df["id"].apply(lambda x: f"R{x:0{digit}d}")

    # sort rows
    assert df["id"].is_unique, "id not unique"
    df.sort_values("id", inplace=True, ignore_index=True)

    # sort cols
    c1 = [c for c in df.columns if not c.startswith("meta")]
    c2 = [c for c in df.columns if c.startswith("meta")]
    cols = sorted(c1) + sorted(c2)
    df = df[cols].reset_index(drop=True)
    df.sample(1).iloc[0].to_dict()

    # save
    sroot.DATASET_PQ.parent.mkdir(parents=True, exist_ok=True)
    utils.write_df2(sroot.DATASET_PQ, df)


def main() -> None:
    gen_dataset_file()  # 6.0M, b09c7441, 6000


if __name__ == "__main__":
    tqdm.pandas()
    pandarallel.initialize(progress_bar=True)
    #
    if hasattr(sys, "ps1"):
        pretty.install()
        reload(utils)
        reload(sroot)
    else:
        with logger.catch(onerror=lambda _: sys.exit(1)):
            # python -m src.eval.ner1.prepare_inference
            typer.run(main)
