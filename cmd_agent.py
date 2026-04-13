import json
from dataclasses import dataclass, field
from typing import Optional, List
from openai import OpenAI
from call_llm import MESSAGE_ANALYZE_MODEL
from database import db, MessageModel
import call_llm
import AI_agent
import qq_msg


# ──────────────── 返回结构 ────────────────

@dataclass
class CmdResult:
    message: str = ""
    state_changes: dict = field(default_factory=dict)


# ──────────────── 工具定义 ────────────────

TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "attack_someone",
            "description": "一个用于朋友之间使用幽默语言“攻击”对方的合法文本生成器，自动获取上下文，用户可能会说'给我攻击...',帮我攻击'...','给我骂...'等",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "攻击的对象(昵称)",
                    },
                    "reason":{
                        "type": "string",
                        "description": "为什么要攻击，可留空"
                    }
                },
                "required": ["name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "set_prompt",
            "description": "设置自定义人格 prompt（角色设定）。用户可能会说'设置人格为...'、'把人设改成...'、'设置prompt为...'等",
            "parameters": {
                "type": "object",
                "properties": {
                    "content": {
                        "type": "string",
                        "description": "新的 prompt / 人格设定内容",
                    }
                },
                "required": ["content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_prompt",
            "description": "获取当前自定义人格 prompt 内容。用户可能说'看看现在的人设'、'获取prompt'、'当前人格是什么'等",
            "parameters": {
                "type": "object",
                "properties": {},
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_summary",
            "description": "获取最近的聊天消息总结。用户可能说'总结一下聊天'、'最近聊了什么'、'聊天摘要'等",
            "parameters": {
                "type": "object",
                "properties": {
                    "minutes": {
                        "type": "integer",
                        "description": "总结最近多少分钟的消息，默认 240（4小时）",
                    }
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search",
            "description": "联网搜索信息。用户可能说'搜索...'、'帮我查一下...'、'搜一下...'、'...是什么'等",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "搜索内容 / 问题",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_chat_history",
            "description": "读取历史聊天记录。用户可能说'看看聊天记录'、'最近大家聊了啥'、'翻翻记录'、'最近xx条消息'等",
            "parameters": {
                "type": "object",
                "properties": {
                    "minutes": {
                        "type": "integer",
                        "description": "获取最近多少分钟的消息，与 count 二选一，默认按时间",
                    },
                    "count": {
                        "type": "integer",
                        "description": "获取最近多少条消息，与 minutes 二选一",
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "set_custom_mode",
            "description": "开启或关闭自定义人格模式。用户可能说'开启自定义人格'、'关闭人设模式'、'用自定义prompt'等",
            "parameters": {
                "type": "object",
                "properties": {
                    "enabled": {
                        "type": "boolean",
                        "description": "true 开启自定义人格模式，false 关闭",
                    }
                },
                "required": ["enabled"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "set_reply_threshold",
            "description": "设置当前聊天对象（群或私聊）的回复阈值（0~10）。用户可能说'设置阈值为5'、'把回复灵敏度调到3'、'设置回复阈值'、'调整活跃度'等。阈值越低越活跃，越高越安静。说'重置阈值'则恢复默认值",
            "parameters": {
                "type": "object",
                "properties": {
                    "threshold": {
                        "type": "integer",
                        "description": "回复阈值 0~10，reply_score >= 此值才会回复。不提供则表示查询当前值",
                    },
                    "reset": {
                        "type": "boolean",
                        "description": "true 则重置为默认阈值",
                    }
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "toggle_bot",
            "description": "开启或关闭当前聊天对象（群或私聊）的机器人。用户可能说'开机器人'、'关机器人'、'开启机器人'、'关闭机器人'、'开'、'关'、'闭嘴'、'说话'等",
            "parameters": {
                "type": "object",
                "properties": {
                    "enabled": {
                        "type": "boolean",
                        "description": "true 开启机器人，false 关闭机器人",
                    }
                },
                "required": ["enabled"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "bypass_master_detection",
            "description": "临时关闭或恢复主人消息检测（测试用）。关闭后即使有主人消息，机器人也会正常回复。用户可能说'关闭主人检测'、'暂停master检测'、'测试模式'、'恢复主人检测'等",
            "parameters": {
                "type": "object",
                "properties": {
                    "enabled": {
                        "type": "boolean",
                        "description": "true 关闭主人检测（进入测试模式），false 恢复主人检测",
                    },
                    "minutes": {
                        "type": "integer",
                        "description": "自动恢复时间（分钟），默认 30 分钟。仅在 enabled=true 时有效",
                    }
                },
                "required": ["enabled"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "toggle_target",
            "description": "将当前对话的群或个人添加到/移除出机器人的监听目标列表。用户可能说'添加这个群'、'加入监听'、'移除这个群'、'不要监听这个人'等。不需要参数，自动根据当前对话上下文判断",
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["add", "remove"],
                        "description": "add 添加到监听目标，remove 从监听目标移除",
                    }
                },
                "required": ["action"],
            },
        },
    },
]


# ──────────────── 工具函数 ────────────────
def _attack_someone(context,name:str,reason:str,**_kwargs) -> dict:
    group_id = context.get("group_id")
    user_id = context.get("user_id")
    is_group = context.get("is_group", False)
    system_prompt = f"你是一个不爱说话的主人的“嘴替”，根据上下文使用你所能使用幽默并带点讽刺的语言攻击 [{name}] {f',是因为{reason}，主要攻击这一部分' if reason else ''},直接生成不要解释"
    try :
        if is_group and group_id:
            msg_models: List[MessageModel] = db.get_latest_messages_by_count(group_id, True, 50)
        else :
            msg_models: List[MessageModel] = db.get_latest_messages_by_count(user_id, False, 50)
        user_prompt = "\n".join([msg.to_str() for msg in msg_models])
        reply = call_llm.call_chat_complete(system_prompt,user_prompt,3000,1.5)
        if is_group:
            qq_msg.send_group_message(group_id, reply)
        else:
            qq_msg.send_private_message(user_id, reply)
        print(user_prompt)
        return {"result": f"已生成攻击文本并发送给{name}"}
    except Exception as e:
        return{"result":""}

def _tool_set_prompt(content: str, **_kwargs) -> dict:
    """设置自定义人格 prompt"""
    try:
        with open("./prompts/custom_prompt.txt", "w", encoding="utf-8") as f:
            f.write(content)
        return {"result": "自定义人格 prompt 已更新。"}
    except Exception as e:
        return {"result": f"设置失败：{e}"}


def _tool_get_prompt(**_kwargs) -> dict:
    """获取当前自定义人格 prompt"""
    try:
        with open("./prompts/custom_prompt.txt", "r", encoding="utf-8") as f:
            prompt_content = f.read()
        if not prompt_content.strip():
            return {"result": "当前自定义 prompt 为空。"}
        return {"result": f"当前自定义人格 prompt 内容：{prompt_content}"}
    except FileNotFoundError:
        return {"result": "自定义 prompt 文件不存在，尚未设置。"}
    except Exception as e:
        return {"result": f"读取失败：{e}"}


def _tool_get_summary(context: dict, minutes: int = 240, **_kwargs) -> dict:
    """获取聊天总结"""
    try:
        group_id = context.get("group_id")
        user_id = context.get("user_id")
        is_group = context.get("is_group", False)

        if is_group and group_id:
            msg_models: List[MessageModel] = db.get_latest_messages_by_time(group_id, True, minutes)
        elif user_id:
            msg_models: List[MessageModel] = db.get_latest_messages_by_time(user_id, False, minutes)
        else:
            return {"result": "无法确定聊天目标，获取总结失败。"}

        if not msg_models:
            return {"result": f"最近 {minutes} 分钟内没有找到消息记录。"}

        summary = call_llm.call_chat_complete(
            system_prompt="""你是一个总结专家，请基于以下聊天记录生成详细的对话总结，按照时间戳详细总结所有主题，给出话题内容中重要的部分，并给出话题的参与人。\n
            **不要回复聊天消息内容，你要做的是总结聊天内容**。\n
            输出格式：\n
            总结时间段:yyyy:mm:dd:hh:mm ~ yyyy:mm:dd:hh:mm\n
            1.yyyy:mm:dd:hh:mm ~ yyyy:mm:dd:hh:mm :主题标题\n
            详细内容：xxx\n
            参与人：xxx，xxx\n
            2.yyyy:mm:dd:hh:mm ~ yyyy:mm:dd:hh:mm :主题标题\n
            详细内容：xxxx\n
            参与人：xxx，xxx\n
            ...
            """,
            user_prompt="\n".join([msg.to_str() for msg in msg_models]),
        )
        return {"result": f"最近 {minutes} 分钟的聊天总结：\n{summary}"}
    except Exception as e:
        return {"result": f"生成总结失败：{e}"}


def _tool_get_chat_history(context: dict, minutes: int = None, count: int = None, **_kwargs) -> dict:
    """读取历史聊天记录"""
    try:
        group_id = context.get("group_id")
        user_id = context.get("user_id")
        is_group = context.get("is_group", False)

        target_id = group_id if is_group else user_id
        if not target_id:
            return {"result": "无法确定聊天目标。"}

        if count:
            msg_models = db.get_latest_messages_by_count(target_id, is_group, count)
            label = f"最近 {count} 条"
        else:
            minutes = minutes or 60
            msg_models = db.get_latest_messages_by_time(target_id, is_group, minutes)
            label = f"最近 {minutes} 分钟"

        if not msg_models:
            return {"result": f"{label}内没有找到消息记录。"}

        lines = [msg.to_str() for msg in msg_models]
        return {"result": f"{label}的聊天记录（共 {len(lines)} 条）：\n" + "\n".join(lines)}
    except Exception as e:
        return {"result": f"获取聊天记录失败：{e}"}


def _tool_set_custom_mode(enabled: bool, **_kwargs) -> dict:
    """开启/关闭自定义人格模式"""
    status = "已开启" if enabled else "已关闭"
    return {
        "result": f"自定义人格模式{status}。",
        "state_changes": {"use_custom_prompt": enabled},
    }


def _searchbot_fallback(query: str) -> str:
    """使用 searchbot agent 作为搜索后备（处理模块名冲突）"""
    import sys
    import os

    searchbot_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "searchbot")

    # 暂存与 searchbot 冲突的模块（主要是 config）
    stashed = {}
    for name in list(sys.modules):
        if name in ("config", "agent") or name.startswith("tools"):
            stashed[name] = sys.modules.pop(name)

    sys.path.insert(0, searchbot_dir)
    try:
        from agent import chat_sync
        return chat_sync(query)
    finally:
        # 清理 searchbot 加载的模块
        for name in list(sys.modules):
            if name in ("config", "agent") or name.startswith("tools"):
                del sys.modules[name]
        # 恢复原模块
        sys.modules.update(stashed)
        sys.path.remove(searchbot_dir)


def _tool_search(query: str, **_kwargs) -> dict:
    """联网搜索：优先 Gemini grounding，失败时回退到 searchbot agent"""
    raw_result = None

    # 第一优先级：call_search_llm（Gemini + Google Search grounding）
    try:
        raw_result = call_llm.call_search_llm(query)
        if raw_result and raw_result.strip():
            print(f"[Search] Gemini grounding 成功, 结果长度={len(raw_result)}")
        else:
            raw_result = None
            print("[Search] Gemini grounding 返回空结果，尝试 searchbot 后备")
    except Exception as e:
        print(f"[Search] Gemini grounding 失败: {e}，尝试 searchbot 后备")

    # 第二优先级：searchbot agent（deepseek + google search API + 爬虫）
    if not raw_result:
        try:
            raw_result = _searchbot_fallback(query)
            if raw_result and raw_result.strip():
                print(f"[Search] searchbot 后备成功, 结果长度={len(raw_result)}")
            else:
                return {"result": "搜索未返回有效结果，请换个关键词试试。"}
        except Exception as e:
            print(f"[Search] searchbot 后备也失败: {e}")
            return {"result": f"搜索失败：{e}"}

    return {"result": raw_result}


def _tool_set_reply_threshold(context: dict, threshold: int = None, reset: bool = False, **_kwargs) -> dict:
    """设置/查询/重置当前聊天对象的回复阈值"""
    group_id = context.get("group_id")
    user_id = context.get("user_id")
    is_group = context.get("is_group", False)
    target_id = group_id if is_group else user_id
    if not target_id:
        return {"result": "无法确定当前聊天对象。"}

    target_label = f"群 {target_id}" if is_group else f"用户 {target_id}"

    if reset:
        AI_agent.remove_reply_threshold(target_id, is_group)
        default = AI_agent.get_reply_threshold(target_id, is_group)
        return {"result": f"{target_label} 的回复阈值已重置为默认值 {default}。"}

    if threshold is None:
        current = AI_agent.get_reply_threshold(target_id, is_group)
        return {"result": f"{target_label} 当前回复阈值为 {current}（0~10，越低越活跃）。"}

    if not (0 <= threshold <= 10):
        return {"result": "阈值范围为 0~10，请重新设置。"}

    AI_agent.set_reply_threshold(target_id, is_group, threshold)
    return {"result": f"{target_label} 的回复阈值已设置为 {threshold}。"}


def _tool_toggle_bot(context: dict, enabled: bool, **_kwargs) -> dict:
    """开启/关闭当前聊天对象的机器人"""
    group_id = context.get("group_id")
    user_id = context.get("user_id")
    is_group = context.get("is_group", False)
    target_id = group_id if is_group else user_id
    if not target_id:
        return {"result": "无法确定当前聊天对象。"}

    target_label = f"群 {target_id}" if is_group else f"用户 {target_id}"
    status = "已开启" if enabled else "已关闭"
    return {
        "result": f"{target_label} 的机器人{status}。",
        "state_changes": {"toggle_bot": {"target_id": target_id, "enabled": enabled}},
    }


def _tool_bypass_master_detection(enabled: bool, minutes: int = 30, **_kwargs) -> dict:
    """临时关闭/恢复主人消息检测"""
    if enabled:
        return {
            "result": f"主人消息检测已关闭（测试模式），将在 {minutes} 分钟后自动恢复。",
            "state_changes": {"bypass_master_detection": {"enabled": True, "minutes": minutes}},
        }
    else:
        return {
            "result": "主人消息检测已恢复。",
            "state_changes": {"bypass_master_detection": {"enabled": False}},
        }


def _reload_msg_server_targets():
    """通知 msg_server 热重载监听目标"""
    import requests
    try:
        resp = requests.post("http://127.0.0.1:5002/reload_targets", timeout=5)
        return resp.json().get("status") == "ok"
    except Exception as e:
        print(f"[toggle_target] 通知 msg_server 重载失败: {e}")
        return False


def _tool_toggle_target(context: dict, action: str, **_kwargs) -> dict:
    """添加/移除当前对话到监听目标列表"""
    import os
    group_id = context.get("group_id")
    user_id = context.get("user_id")
    is_group = context.get("is_group", False)

    if is_group:
        target_id = group_id
        target_key = "target_groups"
        label = f"群 {target_id}"
    else:
        target_id = user_id
        target_key = "target_users"
        label = f"用户 {target_id}"

    if not target_id:
        return {"result": "无法确定当前对话目标。"}

    from config import CONFIG, _load_config, _strip_json_comments
    import json, re

    # 读取原始 config.json（带注释）
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")
    with open(config_path, "r", encoding="utf-8") as f:
        raw_text = f.read()

    # 解析为干净 JSON 用于检查
    clean_config = json.loads(_strip_json_comments(raw_text))

    chatbot_targets = clean_config["chatbot_server"][target_key]
    msg_targets = clean_config["message_server"][target_key]

    if action == "add":
        if target_id in chatbot_targets and target_id in msg_targets:
            return {"result": f"{label} 已经在监听目标中。"}

        # 在 chatbot_server 和 message_server 的 target 数组中添加
        for section in ["chatbot_server", "message_server"]:
            section_targets = clean_config[section][target_key]
            if target_id not in section_targets:
                # 找到对应 section 的 target_key 数组的最后一个元素位置，在其后插入
                pattern = rf'("{section}"[\s\S]*?"{target_key}"\s*:\s*\[)([\s\S]*?)(\])'
                match = re.search(pattern, raw_text)
                if match:
                    array_content = match.group(2).rstrip()
                    # 在数组末尾添加
                    if array_content.rstrip().rstrip(",").strip():
                        new_array = array_content.rstrip() + f",\n            {target_id}"
                    else:
                        new_array = f"\n            {target_id}"
                    raw_text = raw_text[:match.start(2)] + new_array + raw_text[match.start(3):]

        # 写回文件
        with open(config_path, "w", encoding="utf-8") as f:
            f.write(raw_text)

        # 更新内存中的 CONFIG
        if target_id not in CONFIG["chatbot_server"][target_key]:
            CONFIG["chatbot_server"][target_key].append(target_id)
        if target_id not in CONFIG["message_server"][target_key]:
            CONFIG["message_server"][target_key].append(target_id)

        _reload_msg_server_targets()
        return {"result": f"{label} 已添加到监听目标。"}

    elif action == "remove":
        if target_id not in chatbot_targets and target_id not in msg_targets:
            return {"result": f"{label} 不在监听目标中。"}

        # 从数组中移除（正则匹配数字和可能的注释、逗号）
        for section in ["chatbot_server", "message_server"]:
            pattern = rf'("{section}"[\s\S]*?"{target_key}"\s*:\s*\[)([\s\S]*?)(\])'
            match = re.search(pattern, raw_text)
            if match:
                array_content = match.group(2)
                # 移除该 target_id 所在的行（包括可能的注释和逗号）
                line_pattern = rf',?\s*\n?\s*{target_id}\s*(?://[^\n]*)?,?'
                new_array = re.sub(line_pattern, '', array_content)
                # 清理可能残留的开头逗号
                new_array = re.sub(r'^\s*,', '', new_array)
                raw_text = raw_text[:match.start(2)] + new_array + raw_text[match.start(3):]

        with open(config_path, "w", encoding="utf-8") as f:
            f.write(raw_text)

        # 更新内存中的 CONFIG
        if target_id in CONFIG["chatbot_server"][target_key]:
            CONFIG["chatbot_server"][target_key].remove(target_id)
        if target_id in CONFIG["message_server"][target_key]:
            CONFIG["message_server"][target_key].remove(target_id)

        _reload_msg_server_targets()
        return {"result": f"{label} 已从监听目标移除。"}

    return {"result": f"未知操作: {action}"}


TOOL_FUNCTIONS = {
    "set_prompt": _tool_set_prompt,
    "get_prompt": _tool_get_prompt,
    "get_summary": _tool_get_summary,
    "get_chat_history": _tool_get_chat_history,
    "search": _tool_search,
    "set_custom_mode": _tool_set_custom_mode,
    "attack_someone":_attack_someone,
    "set_reply_threshold":_tool_set_reply_threshold,
    "toggle_bot":_tool_toggle_bot,
    "bypass_master_detection":_tool_bypass_master_detection,
    "toggle_target":_tool_toggle_target,
}

# 获取类工具：结果直接返回给用户，不经过 LLM 二次处理
DIRECT_RETURN_TOOLS = {"get_prompt", "get_summary", "search","attack_someone"}


# ──────────────── System Prompt ────────────────

SYSTEM_PROMPT = """你是一个QQ聊天机器人的管理助手。用户通过自然语言给你下达指令，你需要理解意图并调用合适的工具来完成任务。

规则：
1. 直接调用工具执行，不要反问确认
2. 执行完后给出简洁的中文结果说明
3. 如果用户的意图不在你的工具能力范围内，直接说明你能做什么

你可以使用的功能：
- 设置/获取自定义人格 prompt
- 获取聊天消息总结
- 读取历史聊天记录
- 开启/关闭自定义人格模式
- 联网搜索信息
- 使用语言攻击工具
- 设置/查询/重置当前聊天的回复阈值（活跃度）
- 开启/关闭当前聊天对象的机器人
- 临时关闭/恢复主人消息检测（测试模式）
- 添加/移除当前对话到机器人监听目标
"""


# ──────────────── Agent 核心 ────────────────

class CmdAgent:
    """命令 Agent：通过 LLM function calling 理解自然语言指令并执行工具"""

    MAX_TURNS = 10

    def __init__(self):
        self.client = OpenAI(
            api_key=MESSAGE_ANALYZE_MODEL.API_KEY,
            base_url=MESSAGE_ANALYZE_MODEL.BASE_URL,
        )
        self.model = MESSAGE_ANALYZE_MODEL.MODEL

    def run(self, user_input: str, context: Optional[dict] = None) -> CmdResult:
        """
        处理用户自然语言指令，返回 CmdResult。

        Args:
            user_input: 用户输入的自然语言指令（已去掉 / 前缀）
            context: 上下文信息，包含 group_id, user_id, is_group
        """
        if context is None:
            context = {}

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_input},
        ]

        all_state_changes = {}
        attack_result = ""
        for turn in range(self.MAX_TURNS):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    tools=TOOLS_SCHEMA,
                    tool_choice="auto",
                )
            except Exception as e:
                print(f"[CmdAgent] LLM 调用失败: {e}")
                return CmdResult(message=f"命令处理失败：{e}")

            choice = response.choices[0]
            message = choice.message

            # 如果 LLM 请求工具调用
            if message.tool_calls:
                # 将 assistant 消息加入历史
                tool_calls_data = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in message.tool_calls
                ]
                messages.append({
                    "role": "assistant",
                    "content": message.content or "",
                    "tool_calls": tool_calls_data, # type: ignore
                })

                # 执行每个工具
                direct_results = []
                for tc in message.tool_calls:
                    tool_result = self._call_tool(tc.function.name, tc.function.arguments, context)

                    # 收集状态变更
                    if "state_changes" in tool_result:
                        all_state_changes.update(tool_result["state_changes"])

                    result_text = tool_result.get("result", "")

                    # 获取类工具直接返回，不再经过 LLM
                    if tc.function.name in DIRECT_RETURN_TOOLS and result_text:
                        direct_results.append(result_text)

                    if tc.function.name == "attack_someone" and result_text:
                        attack_result = result_text
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": result_text,
                    })

                # 如果有直接返回的结果，立即返回不再走 LLM
                if direct_results:
                    return CmdResult(
                        message="\n\n".join(direct_results),
                        state_changes=all_state_changes,
                    )

                continue

            # 没有工具调用，LLM 直接返回了最终回复
            agent_reply = message.content or ""

            if attack_result != "":
                return CmdResult(
                    message=attack_result,
                    state_changes=all_state_changes,
                )

            return CmdResult(
                message=agent_reply,
                state_changes=all_state_changes,
            )

        # 超过最大轮次
        return CmdResult(
            message="命令处理超时，请简化指令后重试。",
            state_changes=all_state_changes,
            )

    def _call_tool(self, name: str, arguments: str, context: dict) -> dict:
        """执行指定工具，返回结果 dict"""
        func = TOOL_FUNCTIONS.get(name)
        if not func:
            print(f"[CmdAgent] 未知工具: {name}")
            return {"result": f"未知工具：{name}"}

        try:
            args = json.loads(arguments) if arguments else {}
            args["context"] = context
            print(f"[CmdAgent] 执行工具 {name}, 参数: {json.dumps(args, ensure_ascii=False, default=str)}")
            result = func(**args)
            print(f"[CmdAgent] 工具 {name} 返回: {result.get('result', '')[:200]}")
            return result
        except Exception as e:
            print(f"[CmdAgent] 工具 {name} 执行异常: {e}")
            return {"result": f"工具 {name} 执行出错：{e}"}
