import sys
from importlib import reload
from pathlib import Path

import typer
from loguru import logger
from pandarallel import pandarallel
from rich import pretty
from tqdm import tqdm

import src.dataset.mt_eval.root
import src.eval.mt_llm.root as sroot
from src import utils

MODELS_STR = """
anonymous/TowerInstruct-7B-v0.2-CC-QLoRA
anonymous/TowerInstruct-7B-v0.2-KLC-QLoRA
anonymous/TowerInstruct-7B-v0.2-AJD-QLoRA
anonymous/TowerInstruct-7B-v0.2-AJD-CC-QLoRA
anonymous/TowerInstruct-7B-v0.2-AJD-KLC-QLoRA
anonymous/TowerInstruct-7B-v0.2-AJD-KLC-CC-QLoRA
anonymous/TowerInstruct-7B-v0.2-AJD-KLC-CC-NoAug-QLoRA

# anonymous/TowerInstruct-7B-v0.2-AJD-KLC-CC-01to0-QLoRA
# anonymous/TowerInstruct-7B-v0.2-AJD-KLC-CC-01to1-QLoRA

anonymous/TowerInstruct-7B-v0.2-AJD-KLC-CC-1_16to0-QLoRA
anonymous/TowerInstruct-7B-v0.2-AJD-KLC-CC-1_16to1-QLoRA

anonymous/TowerInstruct-7B-v0.2-AJD-KLC-CC-1_32to0-QLoRA
anonymous/TowerInstruct-7B-v0.2-AJD-KLC-CC-1_32to1-QLoRA

anonymous/TowerInstruct-7B-v0.2-AJD-KLC-CC-1_8to0-QLoRA
anonymous/TowerInstruct-7B-v0.2-AJD-KLC-CC-1_8to1-QLoRA

anonymous/TowerInstruct-7B-v0.2-AJD-KLC-CC-1_4to0-QLoRA
anonymous/TowerInstruct-7B-v0.2-AJD-KLC-CC-1_4to1-QLoRA

anonymous/TowerInstruct-7B-v0.2-AJD-KLC-CC-05to0-QLoRA
anonymous/TowerInstruct-7B-v0.2-AJD-KLC-CC-05to1-QLoRA

anonymous/TowerInstruct-7B-v0.2-AJD-KLC-CC-1to0-QLoRA
anonymous/TowerInstruct-7B-v0.2-AJD-KLC-CC-1to1-QLoRA

anonymous/TowerInstruct-7B-v0.2-AJD-KLC-CC-2to0-QLoRA
anonymous/TowerInstruct-7B-v0.2-AJD-KLC-CC-2to1-QLoRA
"""


def gen_dataset_file() -> None:
    # read
    df = utils.read_df(src.dataset.mt_eval.root.EVAL_PQ)
    df.sample(1).iloc[0].to_dict()

    # check
    df.groupby(["lang.src", "lang.tgt", "meta.corpus"]).size()
    """
lang.src  lang.tgt  meta.corpus
cc        ko        niu            6000
                    ocdb           6000
                    wyweb_mt       6000
          zh        niu            6000
                    wyweb_mt       6000
hj        en        ajd            2014
          ko        ajd            6000
                    drri           6000
                    drs            6000
                    klc            6000
ko        en        ajd            1965
    """

    # drop ko-en
    idx = df["lang.src"].eq("ko") & df["lang.tgt"].eq("en")
    df = df[~idx].reset_index(drop=True)

    # drop cols
    dcols = [c for c in df.columns if "level_" in c]
    df.drop(columns=dcols, inplace=True)

    # TEMP! keep only drs, drri
    idx = df["meta.corpus"].isin(["drs", "drri"])
    df = df[idx].reset_index(drop=True)

    # add cols
    df["messages"] = df.progress_apply(  # type: ignore
        lambda x: utils.gen_messages_1to1(
            src_lang=x["lang.src"],
            src_text=x["text.src"],
            tgt_lang=x["lang.tgt"],
        ),
        axis=1,
    )
    x1 = df.sample(1).iloc[0].to_dict()
    x1["messages"]

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


def cmd_download_awq() -> None:
    models = MODELS_STR.strip().split("\n")
    models = [s.strip() for s in models]
    models = [s for s in models if s]
    models = [s for s in models if "#" not in s]
    models = [s.replace("-QLoRA", "-AWQ") for s in models]
    models = sorted(set(models))
    cmds = [
        f"conda run --no-capture-output -n mmm huggingface-cli download {s}"
        for s in models
    ]
    cmd = "\n".join(cmds)
    utils.write_sh(sroot.SCRIPT_DIR / "download_awq.sh", cmd)


def main() -> None:
    gen_dataset_file()  # 4.4M, 15fbb048, 12000
    gen_prev_output_file()  # 113.6M, d77b24cb, 488749
    cmd_download_awq()


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
            # python -m src.eval.mt_llm.prepare_inference
            typer.run(main)
