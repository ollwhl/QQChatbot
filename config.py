import json
import re
import os

def _strip_json_comments(text: str) -> str:
    """去除 JSON 中的 // 行注释（不处理字符串内部的 //）"""
    result = []
    in_string = False
    escape = False
    i = 0
    while i < len(text):
        ch = text[i]
        if escape:
            result.append(ch)
            escape = False
            i += 1
            continue
        if ch == '\\' and in_string:
            result.append(ch)
            escape = True
            i += 1
            continue
        if ch == '"':
            in_string = not in_string
            result.append(ch)
            i += 1
            continue
        if not in_string and ch == '/' and i + 1 < len(text) and text[i + 1] == '/':
            # 跳到行尾
            while i < len(text) and text[i] != '\n':
                i += 1
            continue
        result.append(ch)
        i += 1
    return ''.join(result)


def _load_config(config_path: str = None) -> dict:
    if config_path is None:
        config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")
    with open(config_path, "r", encoding="utf-8") as f:
        raw = f.read()
    cleaned = _strip_json_comments(raw)
    return json.loads(cleaned)


# 全局配置，模块导入时加载一次
CONFIG = _load_config()
