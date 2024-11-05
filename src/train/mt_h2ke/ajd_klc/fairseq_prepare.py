import sys
from importlib import reload
from pathlib import Path

import typer
from loguru import logger
from pandarallel import pandarallel
from rich import pretty
from tqdm import tqdm

import src.tool.train as ttool
import src.train.mt_h2ke.ajd_klc.root as sroot
from src import utils


def truncate(s: str, limit: int = sroot.MAX_LEN - 3) -> str:
    s2 = " ".join(s.strip().split()[:limit])
    return s2


def gen_truncate_file() -> None:
    # read
    df = utils.read_df(sroot.ENCODE_PQ)
    df.sample(1).iloc[0].to_dict()

    # truncate: 512 - 3 (BOS, LANG, EOS)
    if 0:
        x1 = df.sample(1).iloc[0].to_dict()
        s = x1["encode.src"]
        truncate(s=s, limit=10)
        truncate(s=s)
    df["encode.src"] = df["encode.src"].parallel_apply(truncate)
    df["encode.tgt"] = df["encode.tgt"].parallel_apply(truncate)
    df.sample(1).iloc[0].to_dict()

    # save
    utils.write_df2(sroot.TRUNCATE_PQ, df)


def gen_fairseq_text_dir() -> None:
    # read
    df = utils.read_df(sroot.TRUNCATE_PQ)
    df.sample(1).iloc[0].to_dict()

    # check
    d_list = []
    for k, _ in df.groupby(["lang.src", "lang.tgt"]):
        src_lang, tgt_lang = k
        # forward direction
        d1 = f"{src_lang}-{tgt_lang}"
        assert d1 not in d_list, f"duplicated direction: {d1}"
        d_list.append(d1)
        # backward direction
        d2 = f"{tgt_lang}-{src_lang}"
        assert d2 not in d_list, f"duplicated direction: {d2}"
        d_list.append(d2)
    d_list.sort()
    logger.debug(f"{len(d_list)} directions: {d_list}")

    # reset
    utils.reset_dir(sroot.FAIRSEQ_TEXT_DIR)

    # for each: subset, src_lang, tgt_lang
    for k, df1 in df.groupby(["split", "lang.src", "lang.tgt"]):
        # name
        logger.debug(k)
        split, src_lang, tgt_lang = k

        # mkdir
        sroot.FAIRSEQ_TEXT_DIR.mkdir(exist_ok=True, parents=True)

        # str
        src_encode = "\n".join(df1["encode.src"])
        tgt_encode = "\n".join(df1["encode.tgt"])

        # src -> tgt
        utils.write_str(
            sroot.FAIRSEQ_TEXT_DIR / f"{split}.{src_lang}-{tgt_lang}.{src_lang}",
            src_encode,
        )
        utils.write_str(
            sroot.FAIRSEQ_TEXT_DIR / f"{split}.{src_lang}-{tgt_lang}.{tgt_lang}",
            tgt_encode,
        )

        # tgt -> src (bi-directional)
        utils.write_str(
            sroot.FAIRSEQ_TEXT_DIR / f"{split}.{tgt_lang}-{src_lang}.{src_lang}",
            src_encode,
        )
        utils.write_str(
            sroot.FAIRSEQ_TEXT_DIR / f"{split}.{tgt_lang}-{src_lang}.{tgt_lang}",
            tgt_encode,
        )


def cmd_fairseq_preprocess() -> None:
    # https://github.com/facebookresearch/fairseq/blob/main/docs/getting_started.rst#data-pre-processing
    # https://github.com/facebookresearch/fairseq/tree/main/examples/mbart#preprocess-data

    files = sorted(list(sroot.FAIRSEQ_TEXT_DIR.rglob("*.*.*")))
    names = [f.name for f in files]
    lang_pairs = sorted(list(set([s.split(".")[1] for s in names])))
    lang_pairs_tuple = [s.split("-") for s in lang_pairs]
    lang_list = sorted(list(set(utils.flatten(lang_pairs_tuple))))

    sroot.LANG_PAIRS_TXT.parent.mkdir(exist_ok=True, parents=True)
    utils.write_str(sroot.LANG_PAIRS_TXT, ",".join(lang_pairs))
    utils.write_str(sroot.LANG_DICT_TXT, "\n".join(lang_list))

    fname_script = sroot.SCRIPT_DIR / "fairseq_preprocess.sh"
    cmds: list[str] = [
        f"# zsh {fname_script}\n",
        f"# rm -rf {sroot.FAIRSEQ_BIN_DIR} && mkdir -p {sroot.FAIRSEQ_BIN_DIR}\n",
    ]

    for i, (src_lang, tgt_lang) in enumerate(lang_pairs_tuple):
        train_pref, valid_pref, test_pref = [
            sroot.FAIRSEQ_TEXT_DIR / f"{split}.{src_lang}-{tgt_lang}"
            for split in ("train", "valid", "test")
        ]
        i1 = i + 1
        cmd1 = f"""
        echo "[{i1}/{len(lang_pairs)}] {src_lang}-{tgt_lang}"
        fairseq-preprocess \\
            --destdir {sroot.FAIRSEQ_BIN_DIR} \\
            --srcdict {sroot.DATA_DICT_TXT} \\
            --tgtdict {sroot.DATA_DICT_TXT} \\
            --thresholdsrc 0 \\
            --thresholdtgt 0 \\
            --source-lang {src_lang} \\
            --target-lang {tgt_lang} \\
            --trainpref {train_pref} \\
            --validpref {valid_pref} \\
            --testpref {test_pref} \\
            --workers 8 \\
            --seed 42
        """
        cmds.append(cmd1)

    cmd1 = f'apprise -vv -t "Done" -b "fairseq_preprocess" "{utils.DISCORD_URL}"'
    cmds.append(cmd1)

    cmd_all = "\n\n".join(cmds)
    sroot.FAIRSEQ_BIN_DIR.mkdir(exist_ok=True, parents=True)
    sroot.SCRIPT_DIR.mkdir(exist_ok=True, parents=True)
    utils.write_sh(fname_script, cmd_all)

    # tmux
    tname = f"fairseq_preprocess_{sroot.MODEL_DIR.name}"
    cmd_tmux = f"""
    tmux new-session -d -s {tname}
    tmux send-keys -t {tname} "cd {Path.cwd().resolve()} && conda run --no-capture-output -n mmm zsh {fname_script}; if [ \\$? -eq 0 ]; then tmux kill-session -t {tname}; fi" C-m
    # tmux attach-session -t {tname}
    # tmux kill-session -t {tname} || true
    # code {sroot.FAIRSEQ_BIN_DIR}
    """.strip()
    fname_tmux = sroot.SCRIPT_DIR / "fairseq_preprocess_tmux.sh"
    utils.write_sh(fname_tmux, cmd_tmux)


def run_fairseq_preprocess_tmux() -> None:
    fname_tmux = sroot.SCRIPT_DIR / "fairseq_preprocess_tmux.sh"
    assert fname_tmux.is_file(), "script not found"
    utils.reset_dir(sroot.FAIRSEQ_BIN_DIR)
    cmd = f"bash {fname_tmux}"
    utils.subprocess_run(cmd)


def main() -> None:
    # truncate
    gen_truncate_file()  # 497.3M, c1f224f2, 416819

    # preprocess
    gen_fairseq_text_dir()
    logger.debug(utils.folder_size(sroot.FAIRSEQ_TEXT_DIR))  # 1.3G
    cmd_fairseq_preprocess()
    run_fairseq_preprocess_tmux()
    logger.debug(f"tmux attach-session -t fairseq_preprocess_{sroot.MODEL_DIR.name}")

    if 0:
        logger.debug(utils.folder_size(sroot.FAIRSEQ_BIN_DIR))  # 437.7M
        utils.open_code(sroot.FAIRSEQ_BIN_DIR)


if __name__ == "__main__":
    tqdm.pandas()
    pandarallel.initialize(progress_bar=True)
    #
    if hasattr(sys, "ps1"):
        pretty.install()
        reload(utils)
        reload(sroot)
        reload(ttool)
    else:
        with logger.catch(onerror=lambda _: sys.exit(1)):
            # python -m src.train.mt_h2ke.ajd_klc.fairseq_prepare
            typer.run(main)
