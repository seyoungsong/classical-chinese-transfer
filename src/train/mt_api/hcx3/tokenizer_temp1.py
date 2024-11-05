from typing import Any, Dict

import httpx


class CompletionExecutor:
    def __init__(
        self, host: str, api_key: str, api_key_primary_val: str, request_id: str
    ) -> None:
        self._host = host
        self._api_key = api_key
        self._api_key_primary_val = api_key_primary_val
        self._request_id = request_id

    def _send_request(
        self, chat_request: Dict[Any, Any], model_name: str = "HCX-003"
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

    def execute(self, chat_request: Dict[Any, Any]) -> str:
        res = self._send_request(chat_request)
        if res["status"]["code"] == "20000":
            return res["result"]["messages"]  # type: ignore
        else:
            return "Error"


if __name__ == "__main__":
    completion_executor = CompletionExecutor(
        host="clovastudio.apigw.ntruss.com",
        api_key="anonymous",
        api_key_primary_val="anonymous",
        request_id="anonymous",
    )
    chat_request = {"messages": [{"role": "user", "content": "hello"}]}
    response_text = completion_executor.execute(chat_request=chat_request)
    print(chat_request)
    print(response_text)
