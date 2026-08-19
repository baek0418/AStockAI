"""AStockAI 统一大模型调用接口。"""

import json
import os
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from dotenv import load_dotenv


load_dotenv(Path(__file__).parent / ".env", override=False)

DEFAULT_API_URL = "https://api.openai.com/v1/chat/completions"
UNAVAILABLE_MESSAGE = "AI服务暂不可用"


def get_ai_settings():
    """从环境变量读取大模型接口地址、模型名称和 API Key。"""
    return {
        "api_key": os.getenv("AI_API_KEY", "").strip(),
        "api_url": os.getenv("AI_API_URL", DEFAULT_API_URL).strip(),
        "model": os.getenv("AI_MODEL", "").strip(),
    }


def build_request_data(
    prompt,
    model,
    system_prompt=None,
    temperature=None,
    max_tokens=None,
):
    """根据提示词和模型名称创建通用聊天接口的请求数据。"""
    messages = []

    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    messages.append({"role": "user", "content": prompt})

    request_data = {
        "model": model,
        "messages": messages,
    }

    if temperature is not None:
        request_data["temperature"] = temperature

    if max_tokens is not None:
        request_data["max_tokens"] = max_tokens

    return json.dumps(request_data).encode("utf-8")


def get_ai_response(response_data):
    """从通用聊天接口响应中提取 AI 返回的文本内容。"""
    response_json = json.loads(response_data.decode("utf-8"))
    content = response_json["choices"][0]["message"]["content"]

    if not isinstance(content, str):
        raise TypeError("AI 响应 content 不是文本")

    return content.strip()


def get_http_error_message(error):
    """将常见 HTTP 状态码转换为不暴露敏感信息的中文提示。"""
    status_code = error.code

    if status_code in (401, 403):
        message = "身份验证或接口权限失败，请检查 AI_API_KEY 和账户权限。"
    elif status_code == 400:
        message = "请求参数无效，请检查模型名称、接口地址或调用参数。"
    elif status_code == 404:
        message = "接口地址或模型名称不存在，请检查 AI_API_URL 和 AI_MODEL。"
    elif status_code == 429:
        message = "请求过于频繁或账户额度不足，请稍后重试并检查账户额度。"
    elif 500 <= status_code <= 599:
        message = "AI 服务暂时异常，请稍后重试。"
    else:
        message = f"接口返回 HTTP {status_code}，请检查服务状态或接口配置。"

    return f"{UNAVAILABLE_MESSAGE}：{message}"


def send_ai_request(
    prompt,
    settings,
    system_prompt=None,
    temperature=None,
    max_tokens=None,
):
    """向已配置的大模型服务发送请求，并返回 AI 文本或错误信息。"""
    request = Request(
        settings["api_url"],
        data=build_request_data(
            prompt,
            settings["model"],
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        ),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {settings['api_key']}",
        },
        method="POST",
    )

    try:
        with urlopen(request, timeout=30) as response:
            ai_response = get_ai_response(response.read())
    except HTTPError as error:
        return get_http_error_message(error)
    except (URLError, TimeoutError, OSError):
        return f"{UNAVAILABLE_MESSAGE}：网络或接口调用失败。"
    except (json.JSONDecodeError, KeyError, IndexError, AttributeError, TypeError):
        return f"{UNAVAILABLE_MESSAGE}：接口返回内容格式异常。"

    if not ai_response:
        return f"{UNAVAILABLE_MESSAGE}：接口未返回有效内容。"

    return ai_response


def validate_request_options(system_prompt, temperature, max_tokens):
    """校验可选请求参数，避免将错误参数发送给模型接口。"""
    if system_prompt is not None and not isinstance(system_prompt, str):
        return "system_prompt 必须是文本。"

    if isinstance(temperature, bool) or (
        temperature is not None and not isinstance(temperature, (int, float))
    ):
        return "temperature 必须是 0 到 2 之间的数字。"

    if temperature is not None and not 0 <= temperature <= 2:
        return "temperature 必须是 0 到 2 之间的数字。"

    if isinstance(max_tokens, bool) or (
        max_tokens is not None and not isinstance(max_tokens, int)
    ):
        return "max_tokens 必须是正整数。"

    if max_tokens is not None and max_tokens <= 0:
        return "max_tokens 必须是正整数。"

    return None


def call_ai_model(prompt, system_prompt=None, temperature=None, max_tokens=None):
    """调用已配置的大模型，并在不可用时返回明确错误信息。

    保持 ``call_ai_model(prompt)`` 的旧调用方式兼容；调用方可按需传入
    system_prompt、temperature 和 max_tokens 来控制单次请求。
    """
    if not isinstance(prompt, str) or not prompt.strip():
        return f"{UNAVAILABLE_MESSAGE}：提示词不能为空。"

    validation_error = validate_request_options(
        system_prompt,
        temperature,
        max_tokens,
    )
    if validation_error:
        return f"{UNAVAILABLE_MESSAGE}：{validation_error}"

    if os.getenv("ASTOCKAI_DISABLE_AI") == "1":
        return f"{UNAVAILABLE_MESSAGE}：当前运行已禁用 AI。"

    settings = get_ai_settings()

    if not settings["api_key"]:
        return f"{UNAVAILABLE_MESSAGE}：未配置 AI_API_KEY。"

    if not settings["model"]:
        return f"{UNAVAILABLE_MESSAGE}：未配置 AI_MODEL。"

    if not settings["api_url"]:
        return f"{UNAVAILABLE_MESSAGE}：未配置 AI_API_URL。"

    return send_ai_request(
        prompt,
        settings,
        system_prompt=system_prompt.strip() if system_prompt else None,
        temperature=temperature,
        max_tokens=max_tokens,
    )
