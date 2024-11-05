import sys
from importlib import reload
from pathlib import Path

import typer
from loguru import logger
from pandarallel import pandarallel
from rich import pretty
from tqdm import tqdm

import src.dataset.ner.root
import src.eval.ner.root as sroot
import src.tool.eval as etool
from src import utils


def gen_dataset_file() -> None:
    # read
    df = utils.read_df(src.dataset.ner.root.EVAL_PQ)
    df.sample(1).iloc[0].to_dict()

    # check
    df.groupby(["meta.corpus", "split"]).size()

    # add cols
    df["text"] = df["text_xml"].progress_apply(
        lambda x: etool.xml2plaintext(x),
    )
    x1 = df.sample(1).iloc[0].to_dict()
    x1["text"]

    # add id
    assert df["key2"].is_unique
    df.sort_values("key2", inplace=True, ignore_index=True)
    #
    df["id"] = df.index + 1
    digit = len(str(max(df["id"])))
    df["id"] = df["id"].apply(lambda x: f"U{x:0{digit}d}")
    df.sample(1).iloc[0].to_dict()

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


def gen_prev_output_file() -> None:
    # not finished for ner
    # check
    if sroot.OUTPUT2_PQ.is_file():
        fname_output2 = sroot.OUTPUT2_PQ
    else:
        logger.warning(f"not found: {sroot.OUTPUT2_PQ}")
        fname_output2 = Path(str(sroot.OUTPUT2_PQ).replace("_v3", "_v2")).resolve()
        logger.warning(f"using old: {fname_output2}")

    # read
    df = utils.read_df(fname_output2)
    df.sample(1).iloc[0].to_dict()

    # filter
    idx = df["pred.content"].isna()
    if idx.sum() > 0:
        logger.debug(df[idx].sample(1).iloc[0].to_dict()["pred.error"])
        logger.warning(
            f"drop {idx.sum()} rows with null pred.content ({idx.mean():.2%})"
        )
        df = df[~idx].reset_index(drop=True)

    # check
    assert (df.groupby(["key2", "pred.model_id"]).size() == 1).all()
    df.sort_values(["key2", "pred.model_id"], inplace=True, ignore_index=True)

    # check
    df1 = df.drop_duplicates("key2", ignore_index=True)
    df1.groupby(["meta.corpus", "lang.src", "lang.tgt"]).size()

    # empty
    vc1 = df["pred.model_id"].value_counts()
    vc2 = vc1.max() - vc1[vc1 < vc1.max()]
    vc2.sort_values(ascending=False, inplace=True)
    logger.warning(f"MISSING: {vc2}")

    # save
    utils.write_df2(sroot.PREV_OUTPUT_PQ, df)


def main() -> None:
    gen_dataset_file()  # 6.0M, 0b4aa6ca, 6000
    if 0:
        gen_prev_output_file()


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
            # python -m src.eval.ner.prepare_inference
            typer.run(main)
