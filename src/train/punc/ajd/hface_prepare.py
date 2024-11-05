import sys
from importlib import reload

import typer
from datasets import load_dataset
from loguru import logger
from pandarallel import pandarallel
from rich import pretty
from tqdm import tqdm

import src.tool.eval as etool
import src.train.punc.ajd.root as sroot
from src import utils


def gen_hface_file() -> None:
    # read
    df0 = utils.read_df(sroot.DATASET_PQ)
    df = df0.copy()
    label2id: dict[str, str] = utils.read_json(sroot.LABEL2ID_JSON)

    # check
    df.sample(1).iloc[0].to_dict()

    # drop bad samples
    idx = df["text"].str.contains(f"<{utils.NER_PREF}")
    idx.sum()
    idx.mean() * 100
    df = df[~idx].reset_index(drop=True)

    # gen id
    assert df["key2"].is_unique, "key2 is not unique"
    df.rename(columns={"key2": "id"}, inplace=True)
    df.drop(columns=["key"], inplace=True)

    # valid -> validation
    df["split"].value_counts()
    df["split"].replace({"valid": "validation"}, inplace=True)

    if 0:
        x1 = df.sample(1).iloc[0].to_dict()
        s = x1["text"]
        not_punc = utils.NOT_PUNC
        punc_xml = etool.text2punc_xml(s=s, not_punc=not_punc, remove_whites=True)
        etool.xml2plaintext(punc_xml)
        tokens, ner_tags = etool.punc_xml2punc_iob(punc_xml=punc_xml, label2id=label2id)
        punc_xml2 = etool.punc_iob2punc_xml(
            tokens=tokens, ner_tags=ner_tags, label2id=label2id
        )
        if punc_xml != punc_xml2:
            utils.temp_diff(punc_xml, punc_xml2)
        etool.text2punc_iob(s=s, not_punc=not_punc, label2id=label2id)

    # convert
    if 0:
        df = df.sample(frac=0.1).sort_index().reset_index(drop=True)
    temp1: utils.SeriesType = df["text"].parallel_apply(
        lambda x: etool.text2punc_iob(s=x, not_punc=utils.NOT_PUNC, label2id=label2id)
    )
    df["tokens"] = temp1.apply(lambda x: x[0])
    df["ner_tags"] = temp1.apply(lambda x: x[1])
    df.sample(1).iloc[0].to_dict()

    # drop meta
    dcols = [c for c in df.columns if c.startswith("meta")]
    df.drop(columns=dcols, inplace=True)

    # save
    sroot.HFACE_PQ.parent.mkdir(exist_ok=True, parents=True)
    utils.write_df2(sroot.HFACE_PQ, df)


def gen_hface_jsonl_dir() -> None:
    # read
    df = utils.read_df(sroot.HFACE_PQ)
    df.sample(1).iloc[0].to_dict()

    # save
    utils.reset_dir(sroot.HFACE_JSONL2_DIR)
    df_sample = df.sample(frac=0.01, random_state=42).reset_index(drop=True)
    for k, df1 in df_sample.groupby("split"):
        logger.debug(f"{k}: {len(df1)}")
        fname = sroot.HFACE_JSONL2_DIR / f"{k}.json"
        df2 = df1.drop(columns=["split"]).reset_index(drop=True)
        df2.to_json(path_or_buf=fname, orient="records", lines=True, force_ascii=False)
        utils.log_written(fname)

    # save
    utils.reset_dir(sroot.HFACE_JSONL_DIR)
    for k, df1 in df.groupby("split"):
        logger.debug(f"{k}: {len(df1)}")
        fname = sroot.HFACE_JSONL_DIR / f"{k}.json"
        df2 = df1.drop(columns=["split"]).reset_index(drop=True)
        df2.to_json(path_or_buf=fname, orient="records", lines=True, force_ascii=False)
        utils.log_written(fname)


def test_hface_jsonl_dir() -> None:
    logger.debug("sample")
    dss = load_dataset(
        "json",
        data_files={
            k: str(sroot.HFACE_JSONL2_DIR / f"{k}.json")
            for k in ["train", "validation", "test"]
        },
        download_mode="force_redownload",
        verification_mode="no_checks",
    )
    logger.success(dss)
    logger.debug({k: len(dss[k]) for k in dss.keys()})
    #
    logger.debug("all")
    dss = load_dataset(
        "json",
        data_files={
            k: str(sroot.HFACE_JSONL_DIR / f"{k}.json")
            for k in ["train", "validation", "test"]
        },
        download_mode="force_redownload",
        verification_mode="no_checks",
    )
    logger.success(dss)
    logger.debug({k: len(dss[k]) for k in dss.keys()})


def main() -> None:
    gen_hface_file()
    gen_hface_jsonl_dir()
    test_hface_jsonl_dir()  # {'train': 293746, 'validation': 37831, 'test': 38332}
    utils.log_written(sroot.HFACE_PQ)  # 184.0M, a7d00ae6


if __name__ == "__main__":
    tqdm.pandas()
    pandarallel.initialize(progress_bar=True)
    #
    if hasattr(sys, "ps1"):
        pretty.install()
        reload(utils)
        reload(sroot)
        reload(etool)
    else:
        with logger.catch(onerror=lambda _: sys.exit(1)):
            # python -m src.train.punc.ajd.hface_prepare
            typer.run(main)
