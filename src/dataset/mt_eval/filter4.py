if 1:
    import os

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    import sys
    from importlib import reload

    import pandas as pd
    import typer
    from loguru import logger
    from pandarallel import pandarallel
    from rich import pretty
    from tqdm import tqdm
    from transformers import AutoTokenizer, BertTokenizerFast

    import src.dataset.mt_eval.root as sroot
    from src import utils


MAX_SRC_TOKEN_LEN_HJ = 512


def count_tokens(s: str, tokenizer: BertTokenizerFast) -> int:
    tok_len = len(tokenizer(s, add_special_tokens=True)["input_ids"])
    return tok_len


def remove_too_long(df: pd.DataFrame, tokenizer: BertTokenizerFast) -> pd.DataFrame:
    if 0:
        x1 = df.sample(1).iloc[0].to_dict()
        s = x1["text.src"]
        count_tokens(s=s, tokenizer=tokenizer)
        df = df.sample(frac=0.01, random_state=42).reset_index(drop=True)
    df["len.src"] = df["text.src"].parallel_apply(
        lambda x: count_tokens(s=x, tokenizer=tokenizer)
    )
    idx1 = df["len.src"] > MAX_SRC_TOKEN_LEN_HJ
    idx2 = df["lang.src"].isin(["zh", "lzh", "cc", "hj"])
    idx = idx1 & idx2
    idx.mean() * 100  # 0.005
    df = df[~idx].reset_index(drop=True)
    return df


def gen_filter4_file() -> None:
    # read
    df = utils.read_df(sroot.FILTER3_PQ)
    df.sample(1).iloc[0].to_dict()

    # length filtering (bert: 512(hj)->x)
    tokenizer = AutoTokenizer.from_pretrained("SIKU-BERT/sikuroberta")
    df = remove_too_long(df=df, tokenizer=tokenizer)

    # drop cols
    df.drop(columns=["len.src"], inplace=True)

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
    utils.write_df2(sroot.FILTER4_PQ, df)


def main() -> None:
    # length filtering
    gen_filter4_file()  # 111.9M, af7a1328, 395410


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
            # python -m src.dataset.mt_eval.filter4
            typer.run(main)
