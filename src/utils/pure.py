import functools
import json
import math
import platform
import random
import re
import subprocess
import sys
import unicodedata
from collections import Counter
from datetime import datetime
from pathlib import Path
from threading import Event, Thread
from typing import Any, Callable, Collection, TypeVar

import numpy as np
import pytz
import tiktoken
from loguru import logger

_X = TypeVar("_X")
_Y = TypeVar("_Y")


TZ_KST = pytz.timezone("Asia/Seoul")

NER_PREF = "▪"

LANG_CODE = {
    "cc": "Classical Chinese",
    "en": "English",
    "hj": "Hanja",
    "ko": "Korean",
    "lzh": "Classical Chinese",
    "zh": "Modern Chinese",
}


def is_korean_char(c: str) -> bool:
    # Extended Korean characters range in Unicode
    pattern = re.compile(
        r"[\u1100-\u11ff\u3130-\u318f\u3200-\u321e\u3260-\u327f\uffa0-\uffdc\uffe6\uAC00-\uD7A3]"
    )
    return bool(pattern.match(c))


def convert1(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return [convert1(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: convert1(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert1(item) for item in obj]
    elif obj is np.nan:
        return None
    elif isinstance(obj, float) and math.isnan(obj):
        return None
    else:
        return obj


def gen_messages_1to1_full(
    src_lang: str, src_text: str, tgt_lang: str, tgt_text: str
) -> str:
    src_lang_x = LANG_CODE[src_lang]
    tgt_lang_x = LANG_CODE[tgt_lang]
    user_content = f"Translate the following text from {src_lang_x} into {tgt_lang_x}.\n{src_lang_x}: {src_text}\n{tgt_lang_x}: "
    if 0:
        print(user_content)
    messages = [
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": tgt_text},
    ]
    msg1_str = json.dumps(messages, ensure_ascii=False)
    return msg1_str


def gen_messages_1to1(src_lang: str, src_text: str, tgt_lang: str) -> str:
    src_lang_x = LANG_CODE[src_lang]
    tgt_lang_x = LANG_CODE[tgt_lang]
    user_content = f"Translate the following text from {src_lang_x} into {tgt_lang_x}.\n{src_lang_x}: {src_text}\n{tgt_lang_x}: "
    if 0:
        print(user_content)
    messages = [{"role": "user", "content": user_content}]
    msg1_str = json.dumps(messages, ensure_ascii=False)
    return msg1_str


def gen_messages_2to1(
    src_lang: str,
    src_text: str,
    ref_lang: str,
    ref_text: str,
    tgt_lang: str,
) -> str:
    src_lang_x = LANG_CODE[src_lang]
    ref_lang_x = LANG_CODE[ref_lang]
    tgt_lang_x = LANG_CODE[tgt_lang]
    user_content = f"Translate the following text from {src_lang_x} into {tgt_lang_x}, based on the reference translation in {ref_lang_x}.\n{src_lang_x}: {src_text}\n{ref_lang_x}: {ref_text}\n{tgt_lang_x}: "
    if 0:
        print(user_content)
    messages = [{"role": "user", "content": user_content}]
    msg1_str = json.dumps(messages, ensure_ascii=False)
    return msg1_str


def calculate_hcx_pricing(input_tokens: int, output_tokens: int, model: str) -> float:
    #
    hcx3_list = ["HCX-003"]
    input_price: float
    output_price: float
    if model in hcx3_list:
        # 0.005 원/token
        won_per_usd = 1380.28
        input_price = 0.005 * 1_000_000 / won_per_usd  # per 1M tokens
        output_price = input_price
    else:
        raise ValueError(f"model={model} not found")

    one_M = 1_000_000
    total_cost = (
        input_price * input_tokens / one_M + output_price * output_tokens / one_M
    )

    return round(total_cost, 2)


def calculate_openai_pricing(
    input_tokens: int, output_tokens: int, model: str
) -> float:
    # https://openai.com/pricing
    # https://azure.microsoft.com/en-us/pricing/details/cognitive-services/openai-service
    gpt4_preview = [
        "gpt-4-0125-preview",
        "gpt-4-1106-preview",
        "gpt-4-1106-vision-preview",
    ]
    gpt35_0125 = ["gpt-3.5-turbo-0125"]

    input_price: float
    output_price: float
    if model in gpt4_preview:
        input_price = 10  # per 1M tokens
        output_price = 30
    elif model in gpt35_0125:
        input_price = 0.5
        output_price = 1.5
    else:
        raise ValueError(f"model={model} not found")

    one_M = 1_000_000
    total_cost = (
        input_price * input_tokens / one_M + output_price * output_tokens / one_M
    )

    return round(total_cost, 2)


def is_punc_unicode(
    c: str, include_whitespace: bool = True, not_punc: str = ""
) -> bool:
    if c in not_punc:
        return False
    cat = unicodedata.category(c)
    if include_whitespace:
        # vertical line is from wyweb
        out = cat[0] in "PZ" or c in "|\n\t"
    else:
        out = cat[0] == "P"
    return out


def is_punctuated_unicode(s: str) -> bool:
    if s is None:
        return False
    for c in s:
        if is_punc_unicode(c):
            return True
    return False


def chunk_by_classifier(s: str, f: Callable) -> list[dict[str, Any]]:  # type: ignore
    if not s:
        return []

    chunk_list: list[dict[str, Any]] = []
    curr_chunk = {"text": s[0], "label": f(s[0])}

    for c1 in s[1:]:
        label1 = f(c1)
        if label1 == curr_chunk["label"]:
            curr_chunk["text"] += c1
        else:
            chunk_list.append(curr_chunk)
            curr_chunk = {"text": c1, "label": label1}

    chunk_list.append(curr_chunk)

    return chunk_list


def unicode_category_full_name(code: str) -> str:
    # Mapping of Unicode category codes to their full names
    _category_names = {
        "Lu": "Letter, Uppercase",
        "Ll": "Letter, Lowercase",
        "Lt": "Letter, Titlecase",
        "Lm": "Letter, Modifier",
        "Lo": "Letter, Other",
        "Mn": "Mark, Nonspacing",
        "Mc": "Mark, Spacing Combining",
        "Me": "Mark, Enclosing",
        "Nd": "Number, Decimal Digit",
        "Nl": "Number, Letter",
        "No": "Number, Other",
        "Pc": "Punctuation, Connector",
        "Pd": "Punctuation, Dash",
        "Ps": "Punctuation, Open",
        "Pe": "Punctuation, Close",
        "Pi": "Punctuation, Initial quote (may behave like Ps or Pe depending on usage)",
        "Pf": "Punctuation, Final quote (may behave like Ps or Pe depending on usage)",
        "Po": "Punctuation, Other",
        "Sm": "Symbol, Math",
        "Sc": "Symbol, Currency",
        "Sk": "Symbol, Modifier",
        "So": "Symbol, Other",
        "Zs": "Separator, Space",
        "Zl": "Separator, Line",
        "Zp": "Separator, Paragraph",
        "Cc": "Other, Control",
        "Cf": "Other, Format",
        "Cs": "Other, Surrogate",
        "Co": "Other, Private Use",
        "Cn": "Other, Not Assigned (no characters in the Unicode Standard have this property)",
    }

    return _category_names.get(code, "Unknown Category")


def num_tiktoken(s: str, model: str = "gpt-4-0613") -> int:
    """Returns the number of tokens in a text string."""
    if s is None:
        return 0
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        print("Warning: model not found. Using cl100k_base encoding.")
        encoding = tiktoken.get_encoding("cl100k_base")
    num_tokens = len(encoding.encode(s))
    return num_tokens


def num_tiktoken_from_messages(
    messages: list[dict[str, str]], model: str = "gpt-4-0613"
) -> int:
    """Return the number of tokens used by a list of messages."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        print("Warning: model not found. Using cl100k_base encoding.")
        encoding = tiktoken.get_encoding("cl100k_base")
    if model in {
        "gpt-3.5-turbo-0125",
        "gpt-3.5-turbo-0613",
        "gpt-3.5-turbo-16k-0613",
        "gpt-4-0125-preview",
        "gpt-4-0314",
        "gpt-4-0613",
        "gpt-4-32k-0314",
        "gpt-4-32k-0613",
    }:
        tokens_per_message = 3
        tokens_per_name = 1
    elif model == "gpt-3.5-turbo-0301":
        # every message follows <|start|>{role/name}\n{content}<|end|>\n
        tokens_per_message = 4
        tokens_per_name = -1  # if there's a name, the role is omitted
    elif "gpt-3.5-turbo" in model:
        print(
            "Warning: gpt-3.5-turbo may update over time. Returning num tokens assuming gpt-3.5-turbo-0613."
        )
        return num_tiktoken_from_messages(messages, model="gpt-3.5-turbo-0613")
    elif "gpt-4" in model:
        print(
            "Warning: gpt-4 may update over time. Returning num tokens assuming gpt-4-0613."
        )
        return num_tiktoken_from_messages(messages, model="gpt-4-0613")
    else:
        raise NotImplementedError(
            f"""num_tokens_from_messages() is not implemented for model {model}. See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens."""
        )
    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
            if key == "name":
                num_tokens += tokens_per_name
    num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
    return num_tokens


def interact_local() -> None:
    # https://stackoverflow.com/a/1396386
    import code

    code.interact(local=locals())


def is_subset_with_count(s1: str, s2: str) -> bool:
    # Check if all characters and their counts in string1 are in string2
    count1 = Counter(s1)
    count2 = Counter(s2)
    for c, n in count1.items():
        if n > count2[c]:
            return False
    return True


def is_cuda_available() -> bool:
    import torch

    if platform.system() == "Darwin":
        return False
    return torch.cuda.is_available()


def os_name() -> str:
    return platform.system().lower()


def flatten(llx: Collection[Collection[_X]]) -> Collection[_X]:
    return [x for lx in llx for x in lx]


def is_interactive() -> bool:
    # source: https://stackoverflow.com/a/64523765
    return hasattr(sys, "ps1")


def shuffle_list(lx: list[_X], seed: int = 42) -> list[_X]:
    lx2 = random.Random(x=seed).sample(lx, len(lx))
    return lx2


def squeeze_whites(s: str) -> str:
    """Replace all space, tab or newline characters to a single space."""
    s1 = re.sub(r"\s+", " ", s).strip()
    return s1


def remove_whites(s: str) -> str:
    s1 = re.sub(r"\s+", "", s).strip()
    return s1


def subprocess_run(cmd: str) -> None:
    logger.debug(f"[subprocess.run] {cmd}")
    subprocess.run(cmd, shell=True)


def code_diff(f1: Path, f2: Path) -> None:
    cmd = f"code --diff {f1} {f2}"
    subprocess_run(cmd)


def open_file(fname: str | Path) -> None:
    p = Path(fname).resolve()
    if platform.system() == "Darwin":
        cmd = f'open "{p}"'
    else:
        cmd = f'code "{p}"'
    subprocess_run(cmd)


def open_code(fname: str | Path) -> None:
    p = Path(fname).resolve()
    cmd = f'code "{p}"'
    subprocess_run(cmd)


def open_url(url: str) -> None:
    if platform.system() == "Darwin":
        cmd = f'open "{url}"'
        subprocess_run(cmd)
    else:
        logger.info(f"url: {url}")


def sort_dict(d: dict[_X, _Y]) -> dict[_X, _Y]:
    keys = list(d.keys())
    keys.sort()
    return {k: d[k] for k in keys}


def transform_keys(input_dict: dict) -> dict:  # type: ignore
    if not isinstance(input_dict, dict):
        return input_dict

    transformed_dict = {}
    for key, value in input_dict.items():
        # Transform the key if it's a tuple
        new_key = "-".join(key) if isinstance(key, tuple) else key

        # Recursively transform the dictionary
        if isinstance(value, dict):
            transformed_dict[new_key] = transform_keys(value)
        elif isinstance(value, list):
            transformed_dict[new_key] = [  # type: ignore
                transform_keys(item) if isinstance(item, dict) else item
                for item in value
            ]
        else:
            transformed_dict[new_key] = value

    return transformed_dict


def notnull_collection(collection: dict | list) -> dict | list:  # type: ignore
    """Recursively remove `None` values from the collection."""
    if isinstance(collection, dict):
        return {k: notnull_collection(v) for k, v in collection.items() if v}
    elif isinstance(collection, list):
        return [notnull_collection(item) for item in collection if item]
    else:
        return collection


def datetime_kst() -> datetime:
    return datetime.now(TZ_KST)


def set_seed(seed: int = 42) -> None:
    import ctranslate2
    import torch
    import torch.backends.cudnn

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    ctranslate2.set_random_seed(seed=seed)
    logger.debug(f"set_seed: {seed}")


def timeout_deco(seconds_before_timeout: float) -> Callable:  # type: ignore
    def decorator(func: Callable) -> Callable:  # type: ignore
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:  # type: ignore
            result = None
            exception = None

            def new_func(done_event: Event):  # type: ignore
                nonlocal result, exception
                try:
                    result = func(*args, **kwargs)
                except Exception as e:
                    exception = e
                finally:
                    done_event.set()

            done_event = Event()
            t = Thread(target=new_func, args=(done_event,))
            t.start()
            t.join(seconds_before_timeout)
            if not done_event.is_set():
                t.join()
                raise RuntimeError(f"Function {func.__name__} timeout")
            if exception:
                raise exception
            return result

        return wrapper

    return decorator


def error_deco(func: Callable) -> Callable:  # type: ignore
    @functools.wraps(func)
    def wrapper(*args, **kwargs) -> Any:  # type: ignore
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.warning(f"error_deco | func: {func.__name__} | error: {repr(e)}")
            return "Error"

    return wrapper
