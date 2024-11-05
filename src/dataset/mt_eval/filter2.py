if 1:
    import os

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    import json
    import sys
    from importlib import reload

    import pandas as pd
    import typer
    from loguru import logger
    from pandarallel import pandarallel
    from rich import pretty
    from tqdm import tqdm
    from transformers import AutoTokenizer, LlamaTokenizerFast

    import src.dataset.mt_eval.root as sroot
    from src import utils


MODEL_NAME = "Unbabel/TowerInstruct-7B-v0.2"


def count_tokens(msg1_str: str, tokenizer: LlamaTokenizerFast) -> int:
    try:
        messages: list[dict[str, str]] = json.loads(msg1_str)
        token_ids = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=False
        )
        token_num = len(token_ids)
        if 0:
            tokenizer.chat_template
            tokenizer.decode(token_ids)
            tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
        return token_num
    except BaseException:
        logger.warning(f"error in count_tokens_safe: {msg1_str}")
        return -1


def remove_too_long(df: pd.DataFrame, tokenizer: LlamaTokenizerFast) -> pd.DataFrame:
    if 0:
        x1 = df.sample(1).iloc[0].to_dict()
        msg1_str = x1["messages"]
        count_tokens(msg1_str=msg1_str, tokenizer=tokenizer)
        df = df.sample(frac=0.01, random_state=42).reset_index(drop=True)
    df["len"] = df["messages"].parallel_apply(
        lambda x: count_tokens(msg1_str=x, tokenizer=tokenizer)
    )
    idx = (df["len"] > 2048) | (df["len"] <= 0)
    df = df[~idx].reset_index(drop=True)
    return df


def gen_filter2_file() -> None:
    # read
    df = utils.read_df(sroot.FILTER_PQ)
    df.sample(1).iloc[0].to_dict()

    # gen messages
    df["messages"] = df.progress_apply(  # type: ignore
        lambda x: utils.gen_messages_1to1_full(
            src_lang=x["lang.src"],
            src_text=x["text.src"],
            tgt_lang=x["lang.tgt"],
            tgt_text=x["text.tgt"],
        ),
        axis=1,
    )

    # length filtering (tower 2048)
    tokenizer: LlamaTokenizerFast = AutoTokenizer.from_pretrained(MODEL_NAME)
    df = remove_too_long(df=df, tokenizer=tokenizer)

    # drop cols
    df.drop(columns=["len", "messages"], inplace=True)

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
    utils.write_df2(sroot.FILTER2_PQ, df)


def main() -> None:
    # length filtering
    gen_filter2_file()  # 112.0M, 1584e18a, 395464


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
            # python -m src.dataset.mt_eval.filter2
            typer.run(main)
