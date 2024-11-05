import json
import sys
import time
from importlib import reload
from typing import Any

import tenacity
import typer
from loguru import logger
from openai import OpenAI, Stream
from openai.types.chat import ChatCompletion, ChatCompletionChunk
from rich import pretty
from tqdm import tqdm

import src.train.mt_api.gpt4.root as sroot
from src import utils


class HanjaTranslator:
    def __init__(self) -> None:
        self.client = _load_client()

    def chat_str(
        self,
        model_id: str,
        msg1_str: str,
        temperature: float,
        timeout: float,
        stream: bool = False,
        **kwargs: Any,
    ) -> dict[str, Any]:
        return _chat_str(
            model_id=model_id,
            client=self.client,
            msg1_str=msg1_str,
            temperature=temperature,
            timeout=timeout,
            stream=stream,
            **kwargs,
        )


def _load_client() -> OpenAI:
    # https://platform.openai.com/api-keys
    OPENAI_API_KEY = "anonymous"
    client = OpenAI(
        api_key=OPENAI_API_KEY,
    )
    return client


def _before_sleep(retry_state: tenacity.RetryCallState) -> None:
    logger.warning(
        f"Retrying {retry_state.fn}: attempt {retry_state.attempt_number} ended with: {retry_state.outcome}",
    )


@utils.error_deco
@tenacity.retry(wait=tenacity.wait_random(min=2, max=4), stop=tenacity.stop_after_attempt(3), before_sleep=_before_sleep)  # fmt: skip
def _chat_completion_stream(
    model_id: str,
    client: OpenAI,
    msg1_str: str,
    temperature: float,
    timeout: float,
    **kwargs: Any,
) -> dict[str, Any]:
    messages: list[dict[str, str]] = json.loads(msg1_str)
    start_time = time.time()
    completion: Stream[ChatCompletionChunk] = client.chat.completions.create(
        messages=messages,  # type: ignore
        model=model_id,
        temperature=temperature,
        stream=True,
        timeout=timeout,
        **kwargs,
    )
    collected_chunks: list[ChatCompletionChunk] = []
    collected_messages: list[str | None] = []
    try:
        print("[START]", flush=True)
        while True:
            chunk: ChatCompletionChunk = utils.timeout_deco(timeout)(next)(completion)
            collected_chunks.append(chunk)
            if len(chunk.choices) > 0:
                chunk_message = chunk.choices[0].delta.content
            else:
                chunk_message = None
            collected_messages.append(chunk_message)
            if chunk_message is not None:
                print(chunk_message, flush=True, end="")
    except StopIteration:
        print("\n[END]", flush=True)
    duration = round(time.time() - start_time, 2)

    collected_messages2 = [m for m in collected_messages if m is not None]
    content = "".join(collected_messages2)

    finish_reasons = [
        c.choices[0].finish_reason for c in collected_chunks if len(c.choices) > 0
    ]
    finish_reason = [s for s in finish_reasons if s is not None][-1]

    chunk1 = collected_chunks[-1]

    pred: dict[str, Any] = {
        "finish_reason": finish_reason,
        "content": content,
        "model": chunk1.model,
        "model_dump_json": chunk1.model_dump_json(),
        "duration": duration,
        "service": "openai",
        "temperature": temperature,
        "kwargs": kwargs,
    }
    assert isinstance(pred, dict), "pred is not dict 9"
    assert isinstance(pred["content"], str), "pred['content'] is not str 4"
    assert len(pred["content"]) > 0, "pred['content'] is empty 4"

    return pred


@utils.error_deco
@tenacity.retry(wait=tenacity.wait_random(min=2, max=4), stop=tenacity.stop_after_attempt(3), before_sleep=_before_sleep)  # fmt: skip
def _chat_completion(
    model_id: str,
    client: OpenAI,
    msg1_str: str,
    temperature: float,
    timeout: float,
    **kwargs: Any,
) -> dict[str, Any]:
    messages: list[dict[str, str]] = json.loads(msg1_str)
    start_time = time.time()
    completion: ChatCompletion = client.chat.completions.create(
        messages=messages,  # type: ignore
        model=model_id,
        temperature=temperature,
        timeout=timeout,
        **kwargs,
    )
    duration = round(time.time() - start_time, 2)

    pred: dict[str, Any] = {
        "finish_reason": completion.choices[0].finish_reason,
        "content": completion.choices[0].message.content,
        "model": completion.model,
        "model_dump_json": completion.model_dump_json(),
        "duration": duration,
        "service": "openai",
        "temperature": temperature,
        "kwargs": kwargs,
    }
    if completion.usage:
        pred["completion_tokens"] = completion.usage.completion_tokens
        pred["prompt_tokens"] = completion.usage.prompt_tokens
    assert isinstance(pred, dict), "pred is not dict 8"
    assert isinstance(pred["content"], str), "pred['content'] is not str 5"
    assert len(pred["content"]) > 0, "pred['content'] is empty 5"

    return pred


def _chat_str(
    client: OpenAI,
    msg1_str: str,
    model_id: str,
    temperature: float,
    timeout: float,
    stream: bool,
    **kwargs: Any,
) -> dict[str, Any]:
    # inference
    if stream:
        pred = _chat_completion_stream(
            model_id=model_id,
            client=client,
            msg1_str=msg1_str,
            temperature=temperature,
            timeout=timeout,
            **kwargs,
        )
    else:
        pred = _chat_completion(
            model_id=model_id,
            client=client,
            msg1_str=msg1_str,
            temperature=temperature,
            timeout=timeout,
            **kwargs,
        )
    assert isinstance(pred, dict), "pred is not dict 2"
    assert isinstance(pred["content"], str), "pred['content'] is not str 6"
    assert len(pred["content"]) > 0, "pred['content'] is empty 6"

    return pred


def _quick_test() -> None:
    # https://platform.openai.com/docs/models/gpt-4-and-gpt-4-turbo
    model_id = "gpt-4-0125-preview"
    model_id = "gpt-3.5-turbo-0125"

    # setup
    client = _load_client()
    temperature = 0.7
    timeout = 20

    # read
    df = utils.read_df(sroot.DATASET_PQ)
    df.sample(1).iloc[0].to_dict()

    # gen messages
    df["messages"] = df.progress_apply(  # type: ignore
        lambda x: utils.gen_messages_2to1(
            src_lang=x["lang.src"],
            src_text=x["text.src"],
            ref_lang=x["lang.tgt"],
            ref_text=x["text.tgt"],
            tgt_lang="ko",
        ),
        axis=1,
    )

    # sample
    x = df.sample(1).iloc[0].to_dict()
    if 0:
        x["meta.url.cc"]
    msg1_str = x["messages"]
    logger.debug(f"{json.loads(msg1_str)[0]['content']}")
    if 0:
        # filter
        msg1_str = '[{"role": "user", "content": "Translate the following text from Classical Chinese into Korean, based on the reference translation in Modern Chinese.\\nClassical Chinese: 臣请募义征子,率十户一保,愿发山东锐兵六千戍诸州,比五年,蛮可为奴。\\nModern Chinese: 臣请求召募义兵,以十户为一保,再调发山东的精兵六千人卫戍各州。只要五年,蛮人即可为奴。\\nKorean: "}]'
        # safe
        msg1_str = '[{"role": "user", "content": "Translate the following text from Classical Chinese into Korean, based on the reference translation in Modern Chinese.\\nClassical Chinese: 既至中堂,一时崩散。\\nModern Chinese: 到了中堂,元显军一时间崩溃逃散。\\nKorean: "}]'

    # translate
    pred = _chat_str(
        model_id=model_id,
        client=client,
        msg1_str=msg1_str,
        temperature=temperature,
        timeout=timeout,
        stream=False,
    )
    pred = _chat_str(
        model_id=model_id,
        client=client,
        msg1_str=msg1_str,
        temperature=temperature,
        timeout=timeout,
        stream=True,
    )
    utils.temp_diff(x["text.tgt"], pred["content"])
    if 0:
        pred["model_dump"] = json.loads(pred["model_dump_json"])
        utils.write_json(utils.TEMP_JSON, pred)

    # model
    model2 = HanjaTranslator()
    _ = model2.chat_str(
        model_id=model_id,
        msg1_str=msg1_str,
        temperature=temperature,
        timeout=timeout,
    )


def main() -> None:
    _quick_test()


if __name__ == "__main__":
    tqdm.pandas()
    #
    if hasattr(sys, "ps1"):
        pretty.install()
        reload(utils)
        reload(sroot)
    else:
        with logger.catch(onerror=lambda _: sys.exit(1)):
            # python -m src.train.mt_api.gpt4.model_openai
            typer.run(main)
