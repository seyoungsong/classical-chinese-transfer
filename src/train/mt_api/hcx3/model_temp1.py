from typing import Any

import requests


class HyperClovaClient:
    def __init__(
        self, host: str, api_key: str, api_key_primary_val: str, request_id: str
    ) -> None:
        self._host = host
        self._api_key = api_key
        self._api_key_primary_val = api_key_primary_val
        self._request_id = request_id

    def chat_completions_create(self, data: dict[str, Any]) -> dict[Any, Any]:
        headers = {
            "X-NCP-CLOVASTUDIO-API-KEY": self._api_key,
            "X-NCP-APIGW-API-KEY": self._api_key_primary_val,
            "X-NCP-CLOVASTUDIO-REQUEST-ID": self._request_id,
            "Content-Type": "application/json; charset=utf-8",
        }
        r = requests.post(
            self._host + "/testapp/v1/chat-completions/HCX-003",
            headers=headers,
            json=data,
            stream=False,
        )
        r.raise_for_status()
        output: dict[Any, Any] = r.json()
        return output


client = HyperClovaClient(
    host="https://clovastudio.stream.ntruss.com",
    api_key="anonymous",
    api_key_primary_val="anonymous",
    request_id="anonymous",
)


data = {
    "messages": [
        {"role": "user", "content": "사과에 대해서 한 문장으로 설명해 줘."},
    ],
    "topP": 0.8,
    "topK": 0,
    "maxTokens": 256,
    "temperature": 0.5,
    "repeatPenalty": 5.0,
    "stopBefore": ["###"],
    "includeAiFilters": False,
    "seed": 42,
}

output = client.chat_completions_create(data=data)

"""
{
    'status': {'code': '20000', 'message': 'OK'},
    'result': {
        'message': {'role': 'assistant', 'content': '사과는 비타민 C와 식이섬유가 풍부하여 면역력 강화와 피부 미용에 좋은 과일입니다.'},
        'inputLength': 12,
        'outputLength': 23,
        'stopReason': 'stop_before',
        'seed': 42,
        'aiFilter': []
    }
}
"""
