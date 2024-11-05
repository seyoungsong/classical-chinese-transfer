import json
import sys
from importlib import reload
from typing import Any

import typer
from loguru import logger
from rich import pretty
from tqdm import tqdm

import src.train.mt_api.gpt4.model_azure
import src.train.mt_api.gpt4.model_openai
import src.train.mt_api.gpt4.root as sroot
from src import utils


class HanjaTranslator:
    def __init__(self, location: str) -> None:
        self.model_azure = src.train.mt_api.gpt4.model_azure.HanjaTranslator(
            location=location
        )
        self.model_openai = src.train.mt_api.gpt4.model_openai.HanjaTranslator()

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
            model_azure=self.model_azure,
            model_openai=self.model_openai,
            msg1_str=msg1_str,
            temperature=temperature,
            timeout=timeout,
            stream=stream,
            **kwargs,
        )


def _chat_str(
    model_id: str,
    model_azure: src.train.mt_api.gpt4.model_azure.HanjaTranslator,
    model_openai: src.train.mt_api.gpt4.model_openai.HanjaTranslator,
    msg1_str: str,
    temperature: float,
    timeout: float,
    stream: bool,
    **kwargs: Any,
) -> dict[str, Any]:
    # inference
    pred = model_azure.chat_str(
        model_id=model_id,
        msg1_str=msg1_str,
        temperature=temperature,
        timeout=timeout,
        stream=stream,
        **kwargs,
    )
    # fallback if content filtering
    if pred["finish_reason"] == "content_filter":
        logger.warning("content_filter")
        model_dump_json_azure = pred["model_dump_json"]
        pred = model_openai.chat_str(
            model_id=model_id,
            msg1_str=msg1_str,
            temperature=temperature,
            timeout=timeout,
            stream=stream,
            **kwargs,
        )
        pred["model_dump_json_azure"] = model_dump_json_azure

    assert isinstance(pred, dict), "pred is not dict 3"
    assert isinstance(pred["content"], str), "pred['content'] is not str 7"
    assert len(pred["content"]) > 0, "pred['content'] is empty 7"

    return pred


def _quick_test() -> None:
    # setup
    # https://platform.openai.com/docs/models/gpt-4-and-gpt-4-turbo
    model_id = "gpt-4-0125-preview"
    model_id = "gpt-3.5-turbo-0125"
    model_azure = src.train.mt_api.gpt4.model_azure.HanjaTranslator(
        location="northcentralus"
    )
    model_openai = src.train.mt_api.gpt4.model_openai.HanjaTranslator()
    temperature = 0.7
    timeout = 20
    stream = False
    _ = stream

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

    if 0:
        # filter
        msg1_str = '[{"role": "user", "content": "Translate the following text from Classical Chinese into Korean, based on the reference translation in Modern Chinese.\\nClassical Chinese: 臣请募义征子,率十户一保,愿发山东锐兵六千戍诸州,比五年,蛮可为奴。\\nModern Chinese: 臣请求召募义兵,以十户为一保,再调发山东的精兵六千人卫戍各州。只要五年,蛮人即可为奴。\\nKorean: "}]'
        # save
        msg1_str = '[{"role": "user", "content": "Translate the following text from Classical Chinese into Korean, based on the reference translation in Modern Chinese.\\nClassical Chinese: 既至中堂,一时崩散。\\nModern Chinese: 到了中堂,元显军一时间崩溃逃散。\\nKorean: "}]'

    # sample
    x = df.sample(1).iloc[0].to_dict()
    msg1_str = x["messages"]
    logger.debug(f"{json.loads(msg1_str)[0]['content']}")
    # translate
    pred = _chat_str(
        model_id=model_id,
        model_azure=model_azure,
        model_openai=model_openai,
        msg1_str=msg1_str,
        temperature=temperature,
        timeout=timeout,
        stream=False,
    )
    text = json.loads(msg1_str)[0]["content"] + pred["content"]
    print(text)
    pred["model_dump"] = json.loads(pred["model_dump_json"])
    pred.pop("model_dump_json")
    pred["model_dump_azure"] = json.loads(pred["model_dump_json_azure"])
    pred.pop("model_dump_json_azure")
    utils.write_json(utils.TEMP_JSON, {**pred})

    if 0:
        pred = _chat_str(
            model_id=model_id,
            model_azure=model_azure,
            model_openai=model_openai,
            msg1_str=msg1_str,
            temperature=temperature,
            timeout=timeout,
            stream=True,
        )
        pred = {
            "id": "R0879491",
            "key": "魏书/列传/卷一/L0227",
            "key2": "niu_mt|魏书/列传/卷一/L0227|cc|zh",
            "lang.src": "cc",
            "lang.tgt": "zh",
            "messages": '[{"role": "user", "content": "Translate the following text from Classical Chinese into Korean, based on the reference translation in Modern Chinese.\\nClassical Chinese: 未几为乙浑所诛,兄弟皆死。\\nModern Chinese: 不久被乙浑诛杀,兄弟都死了。\\nKorean: "}]',
            "pred.completion_tokens": 28,
            "pred.content": "잠시 후 이웅에 의해 처형당하고 형제 모두 죽다.",
            "pred.duration": 1.1,
            "pred.finish_reason": "stop",
            "pred.model": "gpt-3.5-turbo-0125",
            "pred.model_dump_json": '{"id":"chatcmpl-99wI8Sf6OomL4vtSHe5BaDZtxZP4w","choices":[{"finish_reason":"stop","index":0,"logprobs":null,"message":{"content":"잠시 후 이웅에 의해 처형당하고 형제 모두 죽다.","role":"assistant","function_call":null,"tool_calls":null}}],"created":1712156052,"model":"gpt-3.5-turbo-0125","object":"chat.completion","system_fingerprint":"fp_b28b39ffa8","usage":{"completion_tokens":28,"prompt_tokens":82,"total_tokens":110}}',
            "pred.model_dump_json_azure": '{"id":"chatcmpl-99wI7Wd4oJlp3Xq5PVJ8n4fgR2iwQ","choices":[{"finish_reason":"content_filter","index":0,"logprobs":null,"message":{"content":null,"role":"assistant","function_call":null,"tool_calls":null},"content_filter_results":{"hate":{"filtered":false,"severity":"safe"},"self_harm":{"filtered":false,"severity":"low"},"sexual":{"filtered":false,"severity":"safe"},"violence":{"filtered":true,"severity":"medium"}}}],"created":1712156051,"model":"gpt-35-turbo","object":"chat.completion","system_fingerprint":"fp_2f57f81c11","usage":{"completion_tokens":34,"prompt_tokens":82,"total_tokens":116},"prompt_filter_results":[{"prompt_index":0,"content_filter_results":{"hate":{"filtered":false,"severity":"safe"},"self_harm":{"filtered":false,"severity":"safe"},"sexual":{"filtered":false,"severity":"safe"},"violence":{"filtered":false,"severity":"safe"}}}]}',
            "pred.prompt_tokens": 82,
            "pred.service": "openai",
            "split": "train",
            "text.src": "未几为乙浑所诛,兄弟皆死。",
            "text.tgt": "不久被乙浑诛杀,兄弟都死了。",
            "meta.book_title.cc": "위서(魏书)/열전(列传)/권일(卷一)",
            "meta.corpus": "niu_mt",
            "meta.data_id.cc": "魏书/列传/卷一/L0227",
            "meta.url.cc": "https://github.com/NiuTrans/Classical-Modern/blob/main/双语数据/魏书/列传/卷一/bitext.txt#L679",
        }
        pred["pred.model_dump"] = json.loads(pred["pred.model_dump_json"])
        pred.pop("pred.model_dump_json")
        pred["pred.model_dump_azure"] = json.loads(pred["pred.model_dump_json_azure"])
        pred.pop("pred.model_dump_json_azure")

    if 0:
        utils.temp_diff(x["text.tgt"], pred["content"])

    # model
    model2 = HanjaTranslator(location="northcentralus")
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
            # python -m src.train.mt_api.gpt4.model
            typer.run(main)
