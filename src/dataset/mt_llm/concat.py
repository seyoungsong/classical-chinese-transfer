import json
import sys
from importlib import reload

import pandas as pd
import typer
from loguru import logger
from pandarallel import pandarallel
from rich import pretty
from tqdm import tqdm

import src.crawl.tower_mt.root
import src.dataset.mt_h2ke.root
import src.dataset.mt_llm.root as sroot
from src import utils

"""
<|im_start|>user
Translate the following text from Portuguese into English.
Portuguese: Um grupo de investigadores lançou um novo modelo para tarefas relacionadas com tradução.
English: <|im_end|>
<|im_start|>assistant
A group of researchers has launched a new model for translation-related tasks.
"""


def to_conversations(x1: dict[str, str]) -> str:
    if 0:
        msgs_str = '[{"from": "human", "value": "Translate the following text from Chinese to English.\\nSource: 共享欢乐成为人们旅游休闲的主要目的。\\nReference: "}, {"from": "gpt", "value": "To share happiness has become the main purpose of travel and leisure."}]'
        json.loads(msgs_str)

    src_lang, tgt_lang = x1["lang.src"], x1["lang.tgt"]
    src_text, tgt_text = x1["text.src"], x1["text.tgt"]
    msgs = [
        {
            "from": "human",
            "value": f"Translate the following text from {src_lang} into {tgt_lang}.\n{src_lang}: {src_text}\n{tgt_lang}: ",
        },
        {"from": "gpt", "value": tgt_text},
    ]
    msgs_str = json.dumps(msgs, ensure_ascii=False)
    return msgs_str


def get_mt_h2ke() -> pd.DataFrame:
    # read
    df0 = utils.read_df(src.dataset.mt_h2ke.root.FILTER_PQ)
    df0.sample(1).iloc[0].to_dict()

    # prepare
    df = df0.copy()
    df.sample(1).iloc[0].to_dict()

    # backup
    df["meta.lang.src"] = df["lang.src"]
    df["meta.lang.tgt"] = df["lang.tgt"]

    # replace lang
    cols = ["lang.src", "lang.tgt"]
    {k: k for k in df[cols].stack().unique()}  # type: ignore
    df["lang.src"] = df["lang.src"].replace(utils.LANG_CODE)
    df["lang.tgt"] = df["lang.tgt"].replace(utils.LANG_CODE)
    df.sample(1).iloc[0].to_dict()
    df.groupby(["lang.src", "lang.tgt"]).size()

    # convert to sharegpt style
    if 0:
        x1 = df.sample(1).iloc[0].to_dict()
        json.loads(to_conversations(x1=x1))
        df = df.sample(frac=0.01, random_state=42).reset_index(drop=True)
    df["conversations"] = df.parallel_apply(to_conversations, axis=1)  # type: ignore
    df.sample(1).iloc[0].to_dict()

    # gen cols
    df["lang"] = df["meta.lang.src"] + "-" + df["meta.lang.tgt"]

    # sort rows
    df.sort_values(by=["key2"], inplace=True, ignore_index=True)

    # sort cols
    c1 = [c for c in df.columns if not c.startswith("meta")]
    c2 = [c for c in df.columns if c.startswith("meta")]
    cols = sorted(c1) + sorted(c2)
    df = df[cols].reset_index(drop=True)
    df.sample(1).iloc[0].to_dict()

    # drop cols
    df.sample(1).iloc[0].to_dict()
    cols = ["text.src", "text.tgt", "lang.src", "lang.tgt"]
    df.drop(columns=cols, inplace=True)

    return df


def get_tower_mt() -> pd.DataFrame:
    # read
    df0 = utils.read_df(src.crawl.tower_mt.root.FORMAT2_PQ)
    df0.sample(1).iloc[0].to_dict()

    # prepare
    df = df0.copy()
    df.sample(1).iloc[0].to_dict()

    # sort cols
    c1 = [c for c in df.columns if not c.startswith("meta")]
    c2 = [c for c in df.columns if c.startswith("meta")]
    cols = sorted(c1) + sorted(c2)
    df = df[cols].reset_index(drop=True)
    df.sample(1).iloc[0].to_dict()

    # add key2 for unique sorting
    df["meta.corpus"] = "tower"
    df["key2"] = df["meta.corpus"] + "|" + df["meta.data_id"] + "|" + df["meta.lang"]
    assert df["key2"].is_unique
    df.sample(1).iloc[0].to_dict()

    # rename
    rcols = {"meta.lang": "lang"}
    df.rename(columns=rcols, inplace=True)

    # sort rows
    df.sort_values(by=["key2"], inplace=True, ignore_index=True)

    # sort cols
    c1 = [c for c in df.columns if not c.startswith("meta")]
    c2 = [c for c in df.columns if c.startswith("meta")]
    cols = sorted(c1) + sorted(c2)
    df = df[cols].reset_index(drop=True)
    df.sample(1).iloc[0].to_dict()

    return df


def gen_concat_file() -> None:
    # concat
    df_cat = pd.concat([get_mt_h2ke()], axis=0, ignore_index=True)
    assert df_cat["key2"].is_unique, "key2 is not unique"
    df_cat.sort_values(by=["key2"], inplace=True, ignore_index=True)

    # check
    size = df_cat.groupby(["split", "lang", "meta.corpus"]).size()
    size
    logger.debug(size)
    """
split  lang   meta.corpus
test   cc-ko  niu             96584
              wyweb_mt        19261
       cc-zh  niu             96672
              wyweb_mt        19291
       hj-en  ajd              2123
       hj-ko  ajd             38852
              klc              6895
       ko-en  ajd              2123
train  cc-ko  niu            722560
              wyweb_mt       161086
       cc-zh  niu            723739
              wyweb_mt       161166
       hj-en  ajd             16032
       hj-ko  ajd            299107
              klc             53146
       ko-en  ajd             16012
valid  cc-ko  niu             95834
              wyweb_mt        18271
       cc-zh  niu             95923
              wyweb_mt        18300
       hj-en  ajd              2077
       hj-ko  ajd             38415
              klc              6619
       ko-en  ajd              2077
    """

    # check
    temp1 = df_cat.isna().sum()
    temp1[temp1 > 0]

    # save
    utils.write_df2(sroot.CONCAT_PQ, df_cat)


def main() -> None:
    # concat samples, drop cols, format
    gen_concat_file()  # 638.8M, cc66ca32, 2712165


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
            # python -m src.dataset.mt_llm.concat
            typer.run(main)
