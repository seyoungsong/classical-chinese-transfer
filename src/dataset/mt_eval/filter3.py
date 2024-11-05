if 1:
    import os

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    import sys
    from importlib import reload

    import pandas as pd
    import sentencepiece
    import typer
    from loguru import logger
    from pandarallel import pandarallel
    from rich import pretty
    from tqdm import tqdm

    import src.dataset.mt_eval.root as sroot
    import src.train.mt_h2ke.ajd_klc.root
    from src import utils


MAX_SRC_TOKEN_LEN = 1024
MAX_TGT_TOKEN_LEN = 1024


def count_tokens(s: str, tokenizer: sentencepiece.SentencePieceProcessor) -> int:
    tok_len = len(tokenizer.EncodeAsIds(s)) + 3
    return tok_len


def remove_too_long(
    df: pd.DataFrame, tokenizer: sentencepiece.SentencePieceProcessor
) -> pd.DataFrame:
    if 0:
        x1 = df.sample(1).iloc[0].to_dict()
        text1 = x1["text.src"]
        count_tokens(s=text1, tokenizer=tokenizer)
        df = df.sample(frac=0.01, random_state=42).reset_index(drop=True)
    df["len.src"] = df["text.src"].parallel_apply(
        lambda x: count_tokens(s=x, tokenizer=tokenizer)
    )
    df["len.tgt"] = df["text.tgt"].parallel_apply(
        lambda x: count_tokens(s=x, tokenizer=tokenizer)
    )
    idx1 = df["len.src"] <= MAX_SRC_TOKEN_LEN
    idx2 = df["len.tgt"] <= MAX_TGT_TOKEN_LEN
    idx = idx1 & idx2
    idx.mean() * 100  # 99.99
    df = df[idx].reset_index(drop=True)
    return df


def gen_filter3_file() -> None:
    # read
    df = utils.read_df(sroot.FILTER2_PQ)
    df.sample(1).iloc[0].to_dict()

    # length filtering (h2ke: 1024->1024)
    tokenizer = sentencepiece.SentencePieceProcessor(
        str(src.train.mt_h2ke.ajd_klc.root.CT2_MODEL_DIR / "spm.model")
    )
    df = remove_too_long(df=df, tokenizer=tokenizer)

    # drop cols
    df.drop(columns=["len.src", "len.tgt"], inplace=True)

    # sort rows
    assert df["key2"].is_unique, "key2 is not unique"
    df.sort_values(by=["key2"], inplace=True, ignore_index=True)

    # sort cols
    c1 = [c for c in df.columns if not c.startswith("meta")]
    c2 = [c for c in df.columns if c.startswith("meta")]
    cols = sorted(c1) + sorted(c2)
    df = df[cols].reset_index(drop=True)
    df.sample(1).iloc[0].to_dict()

    # save
    utils.write_df2(sroot.FILTER3_PQ, df)


def main() -> None:
    # length filtering
    gen_filter3_file()  # 111.9M, 6873d325, 395429


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
            # python -m src.dataset.mt_eval.filter3
            typer.run(main)
