import sys
from importlib import reload

import typer
from loguru import logger
from pandarallel import pandarallel
from rich import pretty
from tqdm import tqdm

import src.dataset.punc.root
import src.eval.punc.root as sroot
import src.tool.eval as etool
from src import utils


def gen_dataset_file() -> None:
    # read
    df = utils.read_df(src.dataset.punc.root.EVAL_PQ)
    df.sample(1).iloc[0].to_dict()

    # check
    df.groupby(["meta.corpus", "split"]).size()

    # test
    if 0:
        x = df.sample(1).iloc[0].to_dict()
        s = x["text"]
        not_punc = utils.NOT_PUNC
        s_xml = etool.text2punc_xml(s=s, not_punc=not_punc, remove_whites=True)
        s_nopunc = etool.punc_xml2text_nopunc(s_xml)
        utils.temp_diff(s, s_nopunc)

    # add cols
    df.rename(columns={"text": "meta.text.orig"}, inplace=True)
    df["text_xml"] = df["meta.text.orig"].progress_apply(
        lambda x: etool.text2punc_xml(s=x, not_punc=utils.NOT_PUNC, remove_whites=True)
    )
    df["text"] = df["text_xml"].progress_apply(etool.punc_xml2text_nopunc)
    df.sample(1).iloc[0].to_dict()

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


def main() -> None:
    gen_dataset_file()  # 11.4M, 2028d774, 15000


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
            # python -m src.eval.punc.prepare_inference
            typer.run(main)
