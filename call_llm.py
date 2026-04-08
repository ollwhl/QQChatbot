import os
import requests
import base64
from config import CONFIG


class Model:
    def __init__(self, base_url, api_key, base_model, api_type="openai") -> None:
        self.MODEL = base_model
        self.BASE_URL = base_url
        self.API_KEY = api_key
        self.API_TYPE = api_type


def _build_model(cfg: dict) -> Model:
    """从 config.json 的模型配置段构建 Model 实例"""
    api_key = cfg.get("API_KEY", "")
    if not api_key and cfg.get("API_KEY_ENV"):
        api_key = os.getenv(cfg["API_KEY_ENV"], "")
    return Model(
        base_url=cfg.get("BASE_URL", ""),
        api_key=api_key,
        base_model=cfg.get("MODEL", ""),
        api_type=cfg.get("API_TYPE", "openai"),
    )


_model_cfg = CONFIG["MODEL"]
MESSAGE_ANALYZE_MODEL = _build_model(_model_cfg["MESSAGE_ANALYZE_MODEL"])
CHAT_MODEL = _build_model(_model_cfg["CHAT_MODEL"])
IMAGE_MODEL = _build_model(_model_cfg["IMAGE_MODEL"])
SEARCH_MODEL = _build_model(_model_cfg["SEARCH_MODEL"])

# ──────────────── OpenAI 风格 ────────────────

def _openai_chat(messages: list, use_model: Model, max_tokens: int,
                 temperature: float, timeout: int) -> str:
    payload = {
        "model": use_model.MODEL,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    url = f"{use_model.BASE_URL}/chat/completions"
    headers = {
        "Authorization": f"Bearer {use_model.API_KEY}",
        "Content-Type": "application/json",
    }
    resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    if "choices" not in data or not data["choices"]:
        raise RuntimeError("API 返回了意外的响应格式：没有 choices")
    return data["choices"][0]["message"]["content"]


# ──────────────── Gemini 风格 (google-genai SDK) ────────────────

def _gemini_chat(messages: list, use_model: Model, max_tokens: int,
                 temperature: float, timeout: int) -> str:
    """通过 google-genai 官方 SDK 调用 Gemini"""
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=use_model.API_KEY)

    system_text = None
    contents = []
    for msg in messages:
        role = msg["role"]
        if role == "system":
            system_text = msg["content"] if isinstance(msg["content"], str) else None
            continue
        gemini_role = "user" if role == "user" else "model"
        parts = _to_gemini_parts(msg["content"])
        contents.append(types.Content(role=gemini_role, parts=parts))

    config = types.GenerateContentConfig(
        max_output_tokens=max_tokens,
        temperature=temperature,
        system_instruction=system_text,
        http_options=types.HttpOptions(timeout=timeout * 1000),
    )

    resp = client.models.generate_content(
        model=use_model.MODEL,
        contents=contents,
        config=config,
    )
    if not resp.text:
        raise RuntimeError("Gemini API 返回了空的响应")
    return resp.text


def _to_gemini_parts(content):
    """将 openai 格式的 content 字段转为 google-genai Part 列表"""
    from google.genai import types

    if isinstance(content, str):
        return [types.Part.from_text(text=content)]
    parts = []
    for item in content:
        if item.get("type") == "text":
            parts.append(types.Part.from_text(text=item["text"]))
        elif item.get("type") == "image_url":
            url_or_data = item["image_url"]["url"]
            if url_or_data.startswith("data:"):
                meta, b64data = url_or_data.split(",", 1)
                mime = meta.split(":")[1].split(";")[0]
                parts.append(types.Part.from_bytes(data=base64.b64decode(b64data), mime_type=mime))
            else:
                parts.append(types.Part.from_text(text=f"[图片URL: {url_or_data}]"))
    return parts

def call_search_llm(user_prompt: str) -> str:
    from google import genai
    from google.genai import types
    client = genai.Client()
    grounding_tool = types.Tool(
        google_search=types.GoogleSearch()
    )

    config = types.GenerateContentConfig(
        tools=[grounding_tool]
    )

    response = client.models.generate_content(
        model="gemini-2.5-flash-lite",
        contents=user_prompt,
        config=config,
    )
    return response.text

# ──────────────── Anthropic 风格 (anthropic SDK) ────────────────

def _anthropic_chat(messages: list, use_model: Model, max_tokens: int,
                    temperature: float, timeout: int) -> str:
    """通过 anthropic 官方 SDK 调用 Claude"""
    import anthropic
    import httpx

    client_kwargs = {"api_key": use_model.API_KEY}
    if use_model.BASE_URL:
        client_kwargs["base_url"] = use_model.BASE_URL
    client_kwargs["timeout"] = httpx.Timeout(timeout)
    client = anthropic.Anthropic(**client_kwargs)

    system_text = anthropic.NOT_GIVEN
    api_messages = []
    for msg in messages:
        role = msg["role"]
        if role == "system":
            system_text = msg["content"] if isinstance(msg["content"], str) else anthropic.NOT_GIVEN
            continue
        api_role = "user" if role == "user" else "assistant"
        content = _to_anthropic_content(msg["content"])
        api_messages.append({"role": api_role, "content": content})

    resp = client.messages.create(
        model=use_model.MODEL,
        max_tokens=max_tokens,
        temperature=temperature,
        system=system_text,
        messages=api_messages,
    )
    if not resp.content:
        raise RuntimeError("Anthropic API 返回了空的响应")
    return resp.content[0].text


def _to_anthropic_content(content):
    """将 openai 格式的 content 字段转为 Anthropic SDK 的 content 格式"""
    if isinstance(content, str):
        return content
    blocks = []
    for item in content:
        if item.get("type") == "text":
            blocks.append({"type": "text", "text": item["text"]})
        elif item.get("type") == "image_url":
            url_or_data = item["image_url"]["url"]
            if url_or_data.startswith("data:"):
                meta, b64data = url_or_data.split(",", 1)
                mime = meta.split(":")[1].split(";")[0]
                blocks.append({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": mime,
                        "data": b64data,
                    },
                })
            else:
                blocks.append({"type": "text", "text": f"[图片URL: {url_or_data}]"})
    return blocks


# ──────────────── 统一调度 ────────────────

_DISPATCHERS = {
    "openai": _openai_chat,
    "gemini": _gemini_chat,
    "anthropic": _anthropic_chat,
}


def _dispatch(messages: list, use_model: Model, max_tokens: int,
              temperature: float, timeout: int) -> str:
    handler = _DISPATCHERS.get(use_model.API_TYPE)
    if handler is None:
        raise ValueError(f"不支持的 API_TYPE: {use_model.API_TYPE}")
    return handler(messages, use_model, max_tokens, temperature, timeout)


# ──────────────── 公开接口 ────────────────

def call_chat_complete(system_prompt: str, user_prompt: str, max_tokens: int = 8000,
                       temperature: float = 0.25, timeout: int = 180, use_model: Model = CHAT_MODEL) -> str:
    """
    调用 LLM Chat API（根据 use_model.API_TYPE 自动选择请求格式）
    返回模型的回复文本
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    return _dispatch(messages, use_model, max_tokens, temperature, timeout)


def describe_image(image_url: str, prompt: str = "用中文简洁地描述这张图片的内容，包括人物，文本等细节，力求能让人理解，直接输出描述内容。",
                   timeout: int = 30, use_model: Model = IMAGE_MODEL) -> str:
    """
    使用视觉模型解析图片内容（根据 use_model.API_TYPE 自动选择请求格式）
    """
    use_model = _build_model(_model_cfg["IMAGE_MODEL"])
    if not use_model.API_KEY:
        
        return f"图片（未配置解析） log: use_model:{use_model.MODEL},api_key:{use_model.API_KEY}"

    try:
        # 如果是 URL，先下载图片转 base64（解决 QQ 图片链接无法被 API 服务器访问的问题）
        if image_url.startswith("http"):
            img_headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            }
            img_resp = requests.get(image_url, headers=img_headers, timeout=1)
            img_resp.raise_for_status()

            content_type = img_resp.headers.get('Content-Type', 'image/jpeg')
            if 'png' in content_type:
                media_type = 'image/png'
            elif 'gif' in content_type:
                media_type = 'image/gif'
            elif 'webp' in content_type:
                media_type = 'image/webp'
            else:
                media_type = 'image/jpeg'

            image_data = base64.b64encode(img_resp.content).decode('utf-8')
            image_url = f"data:{media_type};base64,{image_data}"

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": image_url}},
                ],
            }
        ]
        return _dispatch(messages, use_model, max_tokens=300, temperature=0.25, timeout=timeout)

    except requests.Timeout:
        print("图片解析超时")
        return "图片（解析超时）"
    except Exception as e:
        print(f"图片解析失败: {e}")
        return "图片（解析失败）"


if __name__ == '__main__':
    #print(call_search_llm("帮我查查昨天日本选举结果"))
    print(call_chat_complete(system_prompt="扮演爱因斯坦聊天",user_prompt="你好",use_model=MESSAGE_ANALYZE_MODEL))
