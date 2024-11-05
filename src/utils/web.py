import time
from typing import Optional

import apprise
import httpx
import requests
from loguru import logger
from urllib3.exceptions import InsecureRequestWarning

requests.packages.urllib3.disable_warnings(category=InsecureRequestWarning)  # type: ignore


DISCORD_URL = "https://discord.com/api/webhooks/anonymous/anonymous"


_USER_AGENT = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
_USER_AGENT_MOBILE = "Mozilla/5.0 (iPhone; CPU iPhone OS 17_3_1 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.3.1 Mobile/15E148 Safari/604.1"
_HEADERS = {"User-Agent": _USER_AGENT}
_HEADERS_MOBILE = {"User-Agent": _USER_AGENT_MOBILE}
_IP_ADDR: Optional[str] = None


def get_httpx(
    url: str,
    min_len: int = -1,
    max_retry: int = 5,
    timeout: int = 20,
    mobile: bool = False,
) -> str:
    with httpx.Client(verify=False, timeout=timeout) as client:
        for attempt in range(1, max_retry + 1):
            try:
                headers = _HEADERS_MOBILE if mobile else _HEADERS
                response = client.get(url, headers=headers)
                response.raise_for_status()  # Raises exception for 4xx/5xx responses

                if len(response.text) > min_len:
                    return response.text
                else:
                    raise ValueError(
                        f"Response text too small ({len(response.text)} < {min_len})"
                    )

            except (httpx.HTTPStatusError, httpx.RequestError, ValueError) as e:
                _log_and_retry(e, attempt, max_retry, url)

            # Exponential backoff with a maximum
            time.sleep(min(20, 2 ** (attempt - 1)))

    # Log the final failure after all retries
    error_msg = f"Max retry failed for {url}"
    logger.error(error_msg)
    raise RuntimeError(error_msg)


def get_httpx_first_redirect_url(
    url: str, max_retry: int = 5, timeout: int = 20
) -> str:
    with httpx.Client(verify=False, timeout=timeout, follow_redirects=False) as client:
        for attempt in range(1, max_retry + 1):
            try:
                response = client.get(url, headers=_HEADERS)
                if response.is_redirect:
                    return response.headers.get("Location", "")  # type: ignore
                else:
                    return ""

            except (httpx.HTTPStatusError, httpx.RequestError) as e:
                _log_and_retry(e, attempt, max_retry, url)

            # Exponential backoff with a maximum
            time.sleep(min(20, 2 ** (attempt - 1)))

    # Log the final failure after all retries
    error_msg = f"Max retry failed for {url}"
    logger.error(error_msg)
    raise RuntimeError(error_msg)


def _log_and_retry(e: Exception, attempt: int, max_retry: int, url: str) -> None:
    if attempt < max_retry:
        logger.warning(f"Attempt {attempt}/{max_retry} failed for {url}: {e}")
    else:
        logger.error(f"Max retry failed for {url} with the last error: {e}")
        raise RuntimeError(f"Max retry failed for {url} with the last error: {e}")


def requests_get(url: str) -> str:
    max_retry = 3
    for i in range(max_retry):
        try:
            res = requests.get(url=url, timeout=10, headers=_HEADERS, verify=False)
            html = res.text
            return html
        except Exception as e:
            logger.warning(f"Exception | ({i+1}/{max_retry}) | {url} | {repr(e)}")
            time.sleep(1)
            continue
    logger.error(f"Max retry failed: {url}")
    raise RuntimeError(f"Max retry failed: {url}")


def requests_redirect(url: str) -> str:
    max_retry = 3
    for i in range(max_retry):
        try:
            res = requests.get(
                url=url, allow_redirects=False, headers=_HEADERS, verify=False
            )
            url_redirect = res.headers["Location"]
            return url_redirect
        except Exception as e:
            logger.warning(f"Exception | ({i+1}/{max_retry}) | {url} | {repr(e)}")
            time.sleep(1)
            continue
    logger.error(f"Max retry failed: {url}")
    raise RuntimeError(f"Max retry failed: {url}")


def get_ip_addr() -> str:
    # https://api.ipify.org
    # https://ifconfig.co
    # https://ifconfig.io
    # https://ifconfig.me
    # https://ipinfo.io
    global _IP_ADDR
    if _IP_ADDR is not None:
        return _IP_ADDR
    url = "https://ifconfig.io"
    headers = {"User-Agent": "curl/7.64.1", "Accept": "*/*", "Connection": "close"}
    r = httpx.get(url, headers=headers)
    ip_addr = r.text.strip()
    _IP_ADDR = ip_addr
    return ip_addr


def notify(title: str, body: str) -> None:
    # https://github.com/caronc/apprise/wiki/Notify_discord
    apobj = apprise.Apprise()
    apobj.add(DISCORD_URL)
    logger.info(f"notifying: {title}")
    apobj.notify(body=body, title=title)
