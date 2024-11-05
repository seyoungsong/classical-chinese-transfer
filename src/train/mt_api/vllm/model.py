import json
import random
import sys
import time
from importlib import reload
from typing import Any

import httpx
import tenacity
import typer
from loguru import logger
from openai import OpenAI, Stream
from openai.types.chat import ChatCompletion, ChatCompletionChunk
from rich import pretty
from tqdm import tqdm

import src.train.mt_api.vllm.root as sroot
from src import utils

VLLM_RESOURCE_LIST: list[dict[str, str]] = [
    {
        "api_key": "anonymous",
        "base_url": f"https://anonymous.com:1234{i}/v1",
        "gpu": i,  # type: ignore
    }
    for i in [0, 1, 2, 3, 4, 5, 6, 7]
    # for i in [0, 1, 2, 3, 4, 5, 6, 7]
]


class HanjaTranslator:
    def __init__(self, idx: int) -> None:
        self.idx_orig = idx
        self.idx_curr = idx % len(VLLM_RESOURCE_LIST)
        self.resource = _get_resource(idx=self.idx_curr)
        self.client = _load_client(resource=self.resource)

    def chat_str(
        self,
        model_id: str,
        msg1_str: str,
        temperature: float,
        timeout: float,
        stream: bool = False,
        max_try: int = 3,
        **kwargs: Any,
    ) -> dict[str, Any]:
        for num_try in range(max_try):
            try:
                return _chat_str(
                    resource=self.resource,
                    model_id=model_id,
                    client=self.client,
                    msg1_str=msg1_str,
                    temperature=temperature,
                    timeout=timeout,
                    stream=stream,
                    **kwargs,
                )
            except Exception as e:
                logger.error(f"Error | {num_try + 1}/{max_try} | {repr(e)}")
        return _chat_str(
            resource=self.resource,
            model_id=model_id,
            client=self.client,
            msg1_str=msg1_str,
            temperature=temperature,
            timeout=timeout,
            stream=stream,
            **kwargs,
        )


def _get_resource(idx: int) -> dict[str, Any]:
    logger.debug(f"idx: {idx}")
    resource = VLLM_RESOURCE_LIST[idx]
    logger.debug(f"base_url: {resource['base_url']}")
    return resource


def _load_client(resource: dict[str, Any]) -> OpenAI:
    # https://stackoverflow.com/a/77560862
    client = OpenAI(
        api_key=resource["api_key"],
        base_url=resource["base_url"],
        http_client=httpx.Client(http2=True, verify=False),
    )
    return client


def _before_sleep(retry_state: tenacity.RetryCallState) -> None:
    logger.warning(
        f"Retrying {retry_state.fn}: attempt {retry_state.attempt_number} ended with: {retry_state.outcome}",
    )


@tenacity.retry(wait=tenacity.wait_random(min=2, max=4), stop=tenacity.stop_after_attempt(3), before_sleep=_before_sleep)  # fmt: skip
def _chat_completion_stream(
    resource: dict[str, Any],
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
        "base_url": resource["base_url"],
        "service": "vllm",
        "temperature": temperature,
        "kwargs": kwargs,
    }
    assert isinstance(pred, dict), "pred is not dict 6"
    assert isinstance(pred["content"], str), "pred['content'] is not str 1"
    assert len(pred["content"]) > 0, "pred['content'] is empty 1"

    return pred


@tenacity.retry(wait=tenacity.wait_random(min=2, max=4), stop=tenacity.stop_after_attempt(3), before_sleep=_before_sleep)  # fmt: skip
def _chat_completion(
    resource: dict[str, Any],
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
        "base_url": resource["base_url"],
        "service": "vllm",
        "temperature": temperature,
        "kwargs": kwargs,
    }
    if completion.usage:
        pred["completion_tokens"] = completion.usage.completion_tokens
        pred["prompt_tokens"] = completion.usage.prompt_tokens

    # assert isinstance(pred, dict), f"pred is not dict 71 | {pred=}"
    # assert isinstance(pred["content"], str), f"pred['content'] is not str 28 | {pred=}"
    # assert len(pred["content"]) > 0, f"pred['content'] is empty 26 | {pred=}"

    return pred


def _chat_str(
    resource: dict[str, Any],
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
            resource=resource,
            model_id=model_id,
            client=client,
            msg1_str=msg1_str,
            temperature=temperature,
            timeout=timeout,
            **kwargs,
        )
    else:
        pred = _chat_completion(
            resource=resource,
            model_id=model_id,
            client=client,
            msg1_str=msg1_str,
            temperature=temperature,
            timeout=timeout,
            **kwargs,
        )
    assert isinstance(pred, dict), f"pred is not dict 14 | {pred=}"
    assert isinstance(pred["content"], str), f"pred['content'] is not str 3x | {pred=}"
    assert len(pred["content"]) > 0, f"pred['content'] is empty 3v | {pred=}"

    return pred


def _quick_test() -> None:
    """
    Unbabel/TowerInstruct-7B-v0.2
    anonymous/TowerInstruct-7B-v0.2-CC-AWQ
    anonymous/TowerInstruct-7B-v0.2-AJD-AWQ
    anonymous/TowerInstruct-7B-v0.2-AJD-CC-AWQ
    anonymous/TowerInstruct-7B-v0.2-AJD-KLC-AWQ
    anonymous/TowerInstruct-7B-v0.2-AJD-KLC-CC-AWQ
    anonymous/TowerInstruct-7B-v0.2-AJD-KLC-CC-NoAug-AWQ
    """
    model_id = "Unbabel/TowerInstruct-7B-v0.2"

    # setup
    idx = random.choice(range(len(VLLM_RESOURCE_LIST)))
    resource = _get_resource(idx=idx)
    client = _load_client(resource=resource)
    temperature = 0.7
    timeout = 20

    # read
    df = utils.read_df(sroot.DATASET_PQ)
    df.sample(1).iloc[0].to_dict()

    # gen messages
    df["messages"] = df.progress_apply(  # type: ignore
        lambda x: utils.gen_messages_1to1(
            src_lang=x["lang.src"], src_text=x["text.src"], tgt_lang="ko"
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
        resource=resource,
        model_id=model_id,
        client=client,
        msg1_str=msg1_str,
        temperature=temperature,
        timeout=timeout,
        stream=False,
    )
    pred = _chat_str(
        resource=resource,
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
    model2 = HanjaTranslator(idx=0)
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
            # python -m src.train.mt_api.vllm.model
            typer.run(main)
