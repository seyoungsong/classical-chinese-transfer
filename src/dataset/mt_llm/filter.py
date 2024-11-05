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

    import src.dataset.mt_llm.root as sroot
    from src import utils


MODEL_NAME = "Unbabel/TowerInstruct-7B-v0.2"


def dedup_each_split(df: pd.DataFrame) -> pd.DataFrame:
    df_list = [
        df_group.drop_duplicates(subset=["conversations"])
        for _, df_group in df.groupby("split")
    ]
    df1 = pd.concat(df_list, ignore_index=True)
    df1.sort_values(by="key2", inplace=True, ignore_index=True)

    # check
    vc1 = df1["split"].value_counts().sort_index()
    vc0 = df["split"].value_counts().sort_index()
    logger.debug((vc1 / vc0 * 100).sort_index().round(1))

    return df1


def find_dup(train_df: pd.DataFrame, test_df: pd.DataFrame) -> pd.DataFrame:
    merge_df = pd.merge(
        train_df, test_df, on=["conversations"], how="left", indicator=True
    )
    merge_df["_merge"].value_counts()
    merge_df = merge_df[merge_df["_merge"] == "both"].reset_index(drop=True)
    merge_df.rename(columns={"key2_x": "key2"}, inplace=True)
    if 0:
        merge_df.sample(1).iloc[0].to_dict()
    return merge_df


def remove_test_from_train_valid(df: pd.DataFrame) -> pd.DataFrame:
    # check
    df["split"].value_counts()
    assert df["key2"].is_unique, "key2 is not unique"

    # split
    cols = ["conversations", "key2"]
    train_df = df[df["split"] == "train"][cols].reset_index(drop=True)
    valid_df = df[df["split"] == "valid"][cols].reset_index(drop=True)
    test_df = df[df["split"] == "test"][cols].reset_index(drop=True)

    # dedup
    test_df.drop_duplicates(subset=["conversations"], inplace=True, ignore_index=True)

    # find dup
    train_dup = find_dup(train_df=train_df, test_df=test_df)
    valid_dup = find_dup(train_df=valid_df, test_df=test_df)
    final_dup = pd.concat([train_dup, valid_dup], ignore_index=True)

    # check
    assert final_dup["key2"].is_unique
    idx = df["key2"].isin(final_dup["key2"])
    logger.debug(df[idx].groupby("split").size())

    # remove
    df = df[~idx].reset_index(drop=True)

    return df


def remove_valid_from_train(df: pd.DataFrame) -> pd.DataFrame:
    # check
    df["split"].value_counts()
    assert df["key2"].is_unique, "key2 is not unique"

    # split
    cols = ["conversations", "key2"]
    train_df = df[df["split"] == "train"][cols].reset_index(drop=True)
    valid_df = df[df["split"] == "valid"][cols].reset_index(drop=True)

    # dedup
    valid_df.drop_duplicates(subset=["conversations"], inplace=True, ignore_index=True)

    # find dup
    final_dup = find_dup(train_df=train_df, test_df=valid_df)

    # check
    assert final_dup["key2"].is_unique
    idx = df["key2"].isin(final_dup["key2"])
    logger.debug(df[idx].groupby("split").size())

    # remove
    df = df[~idx].reset_index(drop=True)

    return df


def to_messages(conversation: list[dict[str, str]]) -> list[dict[str, str]]:
    key_mapping = {"from": "role", "value": "content"}
    role_mapping = {"human": "user", "gpt": "assistant"}
    if 0:
        d = conversation[0]
        d = {key_mapping[k]: v for k, v in d.items()}
    c1 = [{key_mapping[k]: v for k, v in d.items()} for d in conversation]
    c2 = [{k: role_mapping[v] if k == "role" else v for k, v in d.items()} for d in c1]
    return c2


def count_tokens(msg_str: str, tokenizer: LlamaTokenizerFast) -> int:
    conversation: list[dict[str, str]] = json.loads(msg_str)
    messages = to_messages(conversation=conversation)
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


def count_tokens_safe(msg_str: str, tokenizer: LlamaTokenizerFast) -> int:
    try:
        return count_tokens(msg_str=msg_str, tokenizer=tokenizer)
    except BaseException:
        logger.warning(f"error in count_tokens_safe: {msg_str}")
        return -1


def remove_too_long(df: pd.DataFrame, tokenizer: LlamaTokenizerFast) -> pd.DataFrame:
    if 0:
        x1 = df.sample(1).iloc[0].to_dict()
        msg_str = x1["conversations"]
        count_tokens(msg_str=msg_str, tokenizer=tokenizer)
        df = df.sample(frac=0.01, random_state=42).reset_index(drop=True)
    df["len"] = df["conversations"].parallel_apply(
        lambda x: count_tokens_safe(msg_str=x, tokenizer=tokenizer)
    )
    if 0:
        # check error
        idx = df["len"] <= 0
        idx.sum()
        df[idx].sample(1).iloc[0].to_dict()  # gpt 응답이 null 인 경우 1개.
        df["meta.corpus"].value_counts()

    idx = (df["len"] > 2048) | (df["len"] <= 0)
    if 0:
        (df["len"] > 2048).mean() * 100  # 12.68
        df.groupby("meta.corpus")["len"].describe()
        df[idx].groupby("meta.corpus")["len"].describe()
        df[~idx].groupby("meta.corpus")["len"].describe()
    df = df[~idx].reset_index(drop=True)
    return df


def gen_filter_file() -> None:
    # read
    df0 = utils.read_df(sroot.CONCAT_PQ)
    df = df0.copy()
    df.sample(1).iloc[0].to_dict()

    # dedup each split, remove test from train & valid, remove valid from train
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    df = dedup_each_split(df)
    df = remove_test_from_train_valid(df)
    df = remove_valid_from_train(df)
    df.sort_values(by="key2", inplace=True, ignore_index=True)

    # length filtering (tower 2048)
    tokenizer: LlamaTokenizerFast = AutoTokenizer.from_pretrained(MODEL_NAME)
    df = remove_too_long(df=df, tokenizer=tokenizer)

    # check
    df0["split"].value_counts() / len(df0) * 100
    df["split"].value_counts() / len(df) * 100
    (df["split"].value_counts() / df0["split"].value_counts() * 100).round(
        2
    ).sort_index()
    """
split
train    88.67
valid    88.44
test     83.99
    """

    # save
    utils.write_df2(sroot.FILTER_PQ, df)


def main() -> None:
    # dedup, prevent leakage
    gen_filter_file()  # 372.6M, 26ca0d89, 2643997


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
            # python -m src.dataset.mt_llm.filter
            typer.run(main)
