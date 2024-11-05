import sys
from importlib import reload

import typer
from loguru import logger
from pandarallel import pandarallel
from rich import pretty
from tqdm import tqdm

import src.dataset.mt_aug.root
import src.eval.mt_aug.root as sroot
from src import utils


def gen_dataset_file() -> None:
    # read
    df0 = utils.read_df(src.dataset.mt_aug.root.FILTER_PQ)
    df = df0.copy()
    df.sample(1).iloc[0].to_dict()
    df.columns
    df.info()

    # gen messages
    df["messages"] = df.parallel_apply(  # type: ignore
        lambda x: utils.gen_messages_2to1(
            src_lang=x["lang.src"],
            src_text=x["text.src"],
            ref_lang=x["lang.tgt"],
            ref_text=x["text.tgt"],
            tgt_lang="ko",
        ),
        axis=1,
    )

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
    gen_dataset_file()


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
            # python -m src.eval.mt_aug.prepare_inference
            typer.run(main)
