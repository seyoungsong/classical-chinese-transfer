import sys
from importlib import reload
from typing import Any, Dict

import httpx
import typer
from loguru import logger
from rich import pretty

import src.train.mt_api.hcx3.root as sroot
from src import utils

HOST: str = "clovastudio.apigw.ntruss.com"
API_KEY: str = "anonymous"
API_KEY_PRIMARY_VAL: str = "anonymous"
REQUEST_ID: str = "anonymous"
MODEL_NAME: str = "HCX-003"


class HcxTokenizer:
    def __init__(
        self,
        host: str = HOST,
        api_key: str = API_KEY,
        api_key_primary_val: str = API_KEY_PRIMARY_VAL,
        request_id: str = REQUEST_ID,
    ) -> None:
        self._host = host
        self._api_key = api_key
        self._api_key_primary_val = api_key_primary_val
        self._request_id = request_id

    def _send_request(
        self, chat_request: Dict[Any, Any], model_name: str = MODEL_NAME
    ) -> Dict[str, Any]:
        headers = {
            "Content-Type": "application/json; charset=utf-8",
            "X-NCP-CLOVASTUDIO-API-KEY": self._api_key,
            "X-NCP-APIGW-API-KEY": self._api_key_primary_val,
            "X-NCP-CLOVASTUDIO-REQUEST-ID": self._request_id,
        }

        url = f"https://{self._host}/v1/api-tools/chat-tokenize/{model_name}"

        with httpx.Client() as client:
            response = client.post(url, json=chat_request, headers=headers)

        return response.json()  # type: ignore

    def count_tokens(self, chat_request: Dict[Any, Any]) -> int:
        res = self._send_request(chat_request)
        if res["status"]["code"] == "20000":
            messages: list[dict[str, int]] = res["result"]["messages"]
            total_tokens = sum([d["count"] for d in messages])
            return total_tokens
        else:
            logger.error(f"status code: {res['status']['code']}")
            raise ValueError("Error")


def _quick_test() -> None:
    hcx_tokenizer = HcxTokenizer()
    chat_request = {
        "messages": [
            {"role": "system", "content": "너는 하이브리드 챗봇이야."},
            {"role": "user", "content": "사과에 대해 알려줘"},
        ]
    }
    response_text = hcx_tokenizer.count_tokens(chat_request=chat_request)
    print(chat_request)
    print(response_text)


def main() -> None:
    _quick_test()


if __name__ == "__main__":
    if hasattr(sys, "ps1"):
        pretty.install()
        reload(utils)
        reload(sroot)
    else:
        with logger.catch(onerror=lambda _: sys.exit(1)):
            # python -m src.train.mt_api.hcx3.tokenizer
            typer.run(main)
