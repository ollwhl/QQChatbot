from datetime import datetime
from typing import List, Dict, Optional
from call_llm import call_chat_complete, CHAT_MODEL, MESSAGE_ANALYZE_MODEL, IMAGE_MODEL
import json
import os
import re
from config import CONFIG
from logger import group_chat_logger, private_chat_logger, log_ai_interaction
from database import MessageModel
from memory import PersonaMemory
import threading

# === 人格底座加载 ===
_persona_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "./profile/persona.txt")
_persona_content = ""
try:
    with open(_persona_path, "r", encoding="utf-8") as f:
        _persona_content = f.read()
except FileNotFoundError:
    print(f"Warning: persona.txt not found at {_persona_path}")

# === 回复阈值 ===
_threshold_cfg = CONFIG.get("chatbot_server", {}).get("reply_threshold", {})
_group_reply_threshold = _threshold_cfg.get("default_group", 5)
_private_reply_threshold = _threshold_cfg.get("default_private", 3)
# 从 config 中提取按对象设置的阈值（排除 default_group / default_private）
_config_thresholds = {str(k): v for k, v in _threshold_cfg.items() if k not in ("default_group", "default_private")}

# 运行时动态修改的阈值，持久化到 chat_thresholds.json
_THRESHOLDS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chat_thresholds.json")

def _load_thresholds() -> dict:
    try:
        if os.path.exists(_THRESHOLDS_FILE):
            with open(_THRESHOLDS_FILE, "r", encoding="utf-8") as f:
                return {str(k): v for k, v in json.load(f).items()}
    except Exception:
        pass
    return {}

def _save_thresholds(data: dict):
    with open(_THRESHOLDS_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

_runtime_thresholds = _load_thresholds()

def _make_threshold_key(target_id, is_group: bool) -> str:
    """生成带前缀的阈值 key，如 group_123456 或 private_789012"""
    prefix = "group" if is_group else "private"
    return f"{prefix}_{target_id}"

def set_reply_threshold(target_id, is_group: bool, threshold: int):
    """设置指定聊天对象的回复阈值（运行时，优先级最高）"""
    key = _make_threshold_key(target_id, is_group)
    _runtime_thresholds[key] = threshold
    _save_thresholds(_runtime_thresholds)

def get_reply_threshold(target_id, is_group: bool) -> int:
    """获取阈值，优先级：运行时设置 > config按对象设置 > config默认值"""
    key = _make_threshold_key(target_id, is_group)
    # 优先：运行时通过命令设置的
    runtime = _runtime_thresholds.get(key)
    if runtime is not None:
        return runtime
    # 其次：config.json 中按对象设置的
    config_val = _config_thresholds.get(key)
    if config_val is not None:
        return config_val
    # 最后：全局默认值
    return _group_reply_threshold if is_group else _private_reply_threshold

def remove_reply_threshold(target_id, is_group: bool):
    """移除运行时自定义阈值，恢复使用 config 中的设置"""
    key = _make_threshold_key(target_id, is_group)
    if key in _runtime_thresholds:
        del _runtime_thresholds[key]
        _save_thresholds(_runtime_thresholds)


def extract_json(text: str) -> str:
    """从可能包含 markdown 代码块的文本中提取 JSON"""
    if not text:
        return text

    # 尝试匹配 ```json ... ``` 或 ``` ... ```
    pattern = r'```(?:json)?\s*([\s\S]*?)\s*```'
    match = re.search(pattern, text)
    if match:
        return match.group(1).strip()

    # 如果没有代码块，尝试找到 JSON 对象
    json_pattern = r'\{[\s\S]*\}'
    match = re.search(json_pattern, text)
    if match:
        return match.group(0)

    return text.strip()


def split_messages(reply: str) -> List[str]:
    """将回复按换行符拆分成多条消息"""
    if not reply:
        return []

    # 按换行符拆分
    lines = reply.split('\n')

    # 过滤空行，保留非空消息
    messages = [line.strip() for line in lines if line.strip()]

    return messages if messages else []


def load_prompt(filepath: str) -> str:
    """加载 prompt 文件，替换占位符"""
    with open(filepath, "r", encoding='utf-8') as f:
        content = f.read()
    content = content.replace('$master$', CONFIG['master_name'])
    content = content.replace('$persona$', _persona_content)
    # 分析 prompt 用简短的人格摘要（只取兴趣和性格部分）
    content = content.replace('$persona_summary$', _build_persona_summary())
    return content


def _build_persona_summary() -> str:
    """从完整人格中提取摘要，供分析模型使用"""
    if not _persona_content:
        return ""
    # 提取兴趣和性格相关行，给分析模型一个简短参考
    lines = []
    for line in _persona_content.split('\n'):
        stripped = line.strip()
        if stripped.startswith('-') and any(kw in stripped for kw in ['玩', '追', '兴趣', '爱好', '性格', '风格', '叫我']):
            lines.append(stripped)
    if lines:
        return "【我的简要信息】\n" + "\n".join(lines[:8])
    return ""


def msg_manger(logger, msg_models: List[MessageModel], memory: Optional[PersonaMemory] = None, is_group=True, max_msg=20, use_model=MESSAGE_ANALYZE_MODEL):
    """分析消息，返回 reply_score / topic_summary / reply_to"""
    required_keys = [
        'reply_score',
        'topic_summary',
    ]

    actual_max_msg = min(max_msg, len(msg_models))
    recent_msgs = msg_models[-actual_max_msg:]
    user_prompt = "\n".join([msg.to_str(index=i) for i, msg in enumerate(recent_msgs)])
    system_prompt = ""
    reply = None
    dict_reply = None

    try:
        prompt_file = "./prompts/group_msg_manger_prompt.txt" if is_group else "./prompts/private_msg_manger_prompt.txt"
        system_prompt = load_prompt(prompt_file)

        # 如果有记忆，添加到系统提示中
        if memory and not memory.is_blank:
            memory_context = f"\n【记忆信息】\n{memory.to_str()}"
            system_prompt += memory_context
    except FileNotFoundError as e:
        log_ai_interaction(logger, "msg_manger", system_prompt, user_prompt, "Prompt file not found", "FAILURE", e)
        return None

    try:
        reply = call_chat_complete(system_prompt, user_prompt, 500, 0.0, 120, use_model)
        json_str = extract_json(reply)
        dict_reply = json.loads(json_str)

        missing = [key for key in required_keys if key not in dict_reply]
        if missing:
            raise ValueError(f"Missing required keys: {missing}")

        # 确保 reply_score 是数字
        dict_reply['reply_score'] = int(dict_reply.get('reply_score', 0))
        dict_reply.setdefault('reply_to', '')

        log_ai_interaction(logger, "msg_manger", system_prompt, user_prompt, reply, "SUCCESS")
        return dict_reply
    except Exception as e:
        log_ai_interaction(logger, "msg_manger", system_prompt, user_prompt, reply or "(No reply from AI)", "FAILURE", e)
        return None


def generate_chat_response(logger, context_info: str, msg_models: List[MessageModel], memory: Optional[PersonaMemory], max_msg, reply_to: str = "", is_group: bool = True, use_model=CHAT_MODEL, use_custom_prompt=False):
    """根据分析结果生成聊天回复"""
    system_prompt = ""
    user_prompt = ""
    reply = None

    if not use_custom_prompt:
        try:
            prompt_file = "./prompts/group_chat_response_prompt.txt" if is_group else "./prompts/private_chat_response_prompt.txt"
            system_prompt = load_prompt(prompt_file)
        except FileNotFoundError as e:
            log_ai_interaction(logger, "generate_chat_response", system_prompt, "", "Prompt file not found", "FAILURE", e)
            return None
    else:
        system_prompt = load_prompt("./prompts/custom_prompt.txt")

    try:
        if is_group:
            user_prompt = build_group_user_prompt(
                messages=[msg_model.to_str() for msg_model in msg_models],
                context_info=context_info,
                reply_to=reply_to,
                memory=memory,
                max_messages=max_msg
            )
        else:
            user_prompt = build_private_user_prompt(
                messages=[msg_model.to_str() for msg_model in msg_models],
                context_info=context_info,
                reply_to=reply_to,
                memory=memory,
                max_messages=max_msg
            )

        if user_prompt is None:
            raise ValueError("Failed to build user prompt")

        reply = call_chat_complete(system_prompt, user_prompt, 1000, 1.2, 60, use_model)
        log_ai_interaction(logger, "generate_chat_response", system_prompt, user_prompt, reply, "SUCCESS")
        return reply
    except Exception as e:
        log_ai_interaction(logger, "generate_chat_response", system_prompt, user_prompt, reply or "(No reply from AI)", "FAILURE", e)
        return None


def build_private_user_prompt(
    messages: List[str],
    current_time: Optional[str] = None,
    context_info: Optional[str] = None,
    reply_to: Optional[str] = None,
    memory: Optional[PersonaMemory] = None,
    max_messages: int = 20
) -> str:
    if current_time is None:
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M")

    prompt_parts = []
    prompt_parts.append(f"【当前时间】{current_time}")

    if memory and not memory.is_blank:
        prompt_parts.append(f"\n【记忆信息】\n{memory.to_str()}")

    if context_info:
        prompt_parts.append(f"\n【背景信息】\n{context_info}")

    if reply_to:
        prompt_parts.append(f"\n【回复目标】\n{reply_to}")

    prompt_parts.append("\n【最近的对话消息】")
    recent_messages = messages[-max_messages:] if len(messages) > max_messages else messages
    if not recent_messages:
        prompt_parts.append("(暂无消息记录)")
    else:
        for msg in recent_messages:
            prompt_parts.append(msg)

    prompt_parts.append("下面请开始你的回复")
    return "\n".join(prompt_parts)


def build_group_user_prompt(
    messages: List[str],
    current_time: Optional[str] = None,
    context_info: Optional[str] = None,
    reply_to: Optional[str] = None,
    memory: Optional[PersonaMemory] = None,
    max_messages: int = 20
) -> str:
    """构建群聊 user prompt"""
    if current_time is None:
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    prompt_parts = []
    prompt_parts.append(f"【当前时间】{current_time}")

    if memory and not memory.is_blank:
        prompt_parts.append(f"\n【记忆信息】\n{memory.to_str()}")

    if context_info:
        prompt_parts.append(f"\n【背景信息】\n{context_info}")

    if reply_to:
        prompt_parts.append(f"\n【回复目标】\n{reply_to}")

    prompt_parts.append("\n【最近的群聊消息】")
    recent_messages = messages[-max_messages:] if len(messages) > max_messages else messages
    if not recent_messages:
        prompt_parts.append("(暂无消息记录)")
    else:
        for msg in recent_messages:
            prompt_parts.append(msg)

    return "\n".join(prompt_parts)


def call_AI_agent(logger, msg_models: List[MessageModel], memory: Optional[PersonaMemory], is_group: bool = True, use_custom_prompt: bool = False, target_id=None):
    """
    调用 AI 代理处理消息并生成回复

    Returns:
        str: 单条回复消息
        List[str]: 多条连续回复消息（模拟真人连续发送）
        "": 空字符串表示不需要回复
    """
    if not msg_models:
        logger.warning("No messages to process")
        return ""

    max_msg = min(20, len(msg_models))

    # 第一步：分析消息，得到 reply_score
    analysis = msg_manger(logger, msg_models, memory=memory, is_group=is_group, max_msg=max_msg, use_model=MESSAGE_ANALYZE_MODEL)

    if analysis is None:
        logger.warning("Message analysis failed")
        return ""

    reply_score = analysis.get("reply_score", 0)
    reply_threshold = get_reply_threshold(target_id, is_group) if target_id else (_group_reply_threshold if is_group else _private_reply_threshold)
    logger.info(f"reply_score={reply_score}, threshold={reply_threshold} ({'group' if is_group else 'private'})")

    # 被动记忆：如果 msg_manger 返回了新话题（非空且非null），立即记录到记忆
    summary = analysis.get("topic_summary", "")
    if memory and summary and isinstance(summary, str):
        try:
            memory.remember_topic(summary)
            logger.info(f"记录新话题到记忆: {summary}")
        except Exception as e:
            logger.error(f"记录话题失败: {e}")

    # reply_score 低于阈值，不回复
    if reply_score < reply_threshold:
        logger.info(f"Score {reply_score} < threshold {reply_threshold}, skipping reply.")
        return ""

    # 第二步：构建上下文信息，传给生成模型
    reply_to = analysis.get("reply_to", "")
    reply_to_msg_index = analysis.get("reply_to_msg_index")

    # # 验证 reply_to_msg_index：如果指向的消息之后已有bot回复，说明已处理过，应跳过
    # if reply_to_msg_index is not None and isinstance(reply_to_msg_index, int):
    #     recent_msgs = msg_models[-max_msg:]
    #     if 0 <= reply_to_msg_index < len(recent_msgs):
    #         # 检查该消息之后是否有bot的回复
    #         for i in range(reply_to_msg_index + 1, len(recent_msgs)):
    #             if recent_msgs[i].is_ai or recent_msgs[i].is_master:
    #                 logger.warning(f"reply_to_msg_index={reply_to_msg_index} 指向的消息已被回复过，跳过本次请求")
    #                 return ""

    context_info = f"话题：{summary}" if summary else ""

    # 提取回复目标用户的信息
    target_user_id = None
    target_user_name = None

    if is_group:
        # 群聊：根据 reply_to_msg_index 加载目标用户信息
        if reply_to_msg_index is not None and isinstance(reply_to_msg_index, int):
            recent_msgs = msg_models[-max_msg:]
            if 0 <= reply_to_msg_index < len(recent_msgs):
                target_msg = recent_msgs[reply_to_msg_index]
                target_user_id = target_msg.user_id
                target_user_name = target_msg.sender_card or target_msg.sender_name
    else:
        # 私聊：总是加载对方的信息
        # 从消息中找到对方的 user_id（非 AI、非 master 的发送者）
        for msg in reversed(msg_models):
            if not msg.is_ai and not msg.is_master:
                target_user_id = msg.user_id
                target_user_name = msg.sender_name
                break

        # 如果还是找不到，尝试使用 peer_id
        if target_user_id is None and msg_models:
            last_msg = msg_models[-1]
            if hasattr(last_msg, 'peer_id') and last_msg.peer_id:
                target_user_id = last_msg.peer_id

    # 加载用户信息（profile 和 relationship），不存在则后台生成
    if target_user_id is not None:
        profile_path = f"./profile/{target_user_id}.json"
        relationship_path = f"./relationship/{target_user_id}.json"

        # 检查并后台生成缺失的画像
        need_profile = not os.path.exists(profile_path)
        need_relationship = not os.path.exists(relationship_path)

        if need_profile or need_relationship:
            _gen_group_id = None
            if is_group and msg_models:
                _gen_group_id = getattr(msg_models[0], 'group_id', None)
            _master_user_id = CONFIG.get("master_user_id")
            _gen_target_user_id = target_user_id

            def _bg_generate():
                try:
                    if need_profile:
                        from user_profile import generate_user_profile_llm
                        generate_user_profile_llm(_gen_target_user_id, group_id=_gen_group_id)
                        logger.info(f"后台生成 profile 完成: user_id={_gen_target_user_id}")
                    if need_relationship:
                        from relationship_profile import generate_user_relationship_llm
                        generate_user_relationship_llm(_gen_target_user_id, _master_user_id, group_id=_gen_group_id) # type: ignore
                        logger.info(f"后台生成 relationship 完成: user_id={_gen_target_user_id}")
                except Exception as e:
                    logger.warning(f"后台生成画像失败: user_id={_gen_target_user_id}, error={e}")

            threading.Thread(target=_bg_generate, daemon=True).start()
            logger.info(f"用户 {target_user_id} 缺少画像（profile={need_profile}, relationship={need_relationship}），已启动后台生成")

        user_info_parts = []

        try:
            if os.path.exists(profile_path):
                with open(profile_path, 'r', encoding='utf-8') as f:
                    user_profile = json.load(f)
                    # 提取关键信息
                    interests = user_profile.get("interests", [])
                    personality = user_profile.get("personality", "")
                    comm_style = user_profile.get("communication_style", "")

                    if target_user_name:
                        user_info_parts.append(f"【对话对象：{target_user_name}】" if not is_group else f"【回复对象：{target_user_name}】")
                    if personality:
                        user_info_parts.append(f"性格：{personality[:200]}")
                    if interests:
                        user_info_parts.append(f"兴趣：{', '.join(interests[:5])}")
                    if comm_style:
                        user_info_parts.append(f"沟通风格：{comm_style[:150]}")

            if os.path.exists(relationship_path):
                with open(relationship_path, 'r', encoding='utf-8') as f:
                    user_relationship = json.load(f)
                    # 提取关键关系信息
                    closeness = user_relationship.get("relationship_closeness", "")
                    common_topics = user_relationship.get("common_topics", [])
                    emotional = user_relationship.get("emotional_tendency", "")

                    if closeness:
                        user_info_parts.append(f"亲密度：{closeness}")
                    if common_topics:
                        user_info_parts.append(f"共同话题：{', '.join(common_topics[:3])}")
                    if emotional:
                        user_info_parts.append(f"情感倾向：{emotional[:100]}")

            # 将用户信息添加到 context_info
            if user_info_parts:
                user_info_str = "\n".join(user_info_parts)
                context_info = f"{context_info}\n\n{user_info_str}" if context_info else user_info_str

        except Exception as e:
            logger.warning(f"Failed to load user info for user_id {target_user_id}: {e}")

    reply = generate_chat_response(
        logger, context_info, msg_models, memory, max_msg,
        reply_to=reply_to, is_group=is_group, use_custom_prompt=use_custom_prompt, use_model=CHAT_MODEL
    )

    if not reply:
        logger.info("No reply generated")
        return ""

    if "No reply" in str(reply):
        logger.info("Generator determined no reply is needed.")
        return ""

    # 拆分消息（按换行符）
    messages = split_messages(reply)

    if not messages:
        logger.info("No valid messages after splitting")
        return ""

    logger.info(f"Generated {len(messages)} message(s): {messages}")

    # 第三步：更新记忆（记录bot的回复内容）
    if memory and messages:
        try:
            # 将所有消息合并后记录到记忆中
            combined_reply = ' '.join(messages)
            memory.remember_reply(combined_reply)
            logger.info("Memory updated with new reply.")
        except Exception as e:
            logger.error(f"Failed to update memory: {e}")

    # 返回：单条消息返回字符串，多条消息返回列表
    if len(messages) == 1:
        return messages[0]
    else:
        return messages


def gen_simple_reply(msg_dists: List[str], memory: Optional[PersonaMemory] = None, is_group: bool = True):
    """生成简短回复，支持记忆"""
    logger = group_chat_logger if is_group else private_chat_logger
    required_keys = ['should_reply', 'reply', 'need_more_context']
    max_msg = 20
    max_retry = 3

    system_prompt = ""
    user_prompt = ""
    reply = ""
    reply_json = {}

    try:
        prompt_file = "./simple_group_prompt.txt" if is_group else "./simple_private_prompt.txt"
        system_prompt = load_prompt(prompt_file)

        if memory and not memory.is_blank:
            memory_context = f"\n【记忆信息】\n{memory.to_str()}"
            system_prompt += memory_context
    except FileNotFoundError as e:
        log_ai_interaction(logger, "gen_simple_reply", system_prompt, "", "Prompt file not found", "FAILURE", e)
        return ""

    for attempt in range(max_retry):
        try:
            if is_group:
                user_prompt = build_group_user_prompt(msg_dists, memory=memory, max_messages=max_msg)
            else:
                user_prompt = build_private_user_prompt(msg_dists, memory=memory, max_messages=max_msg)

            reply = call_chat_complete(system_prompt, user_prompt, 500, 1.2, 30, CHAT_MODEL)
            json_str = extract_json(reply)
            reply_json = json.loads(json_str)

            missing = [key for key in required_keys if key not in reply_json]
            if missing:
                raise ValueError(f"Missing required keys: {missing}")

            if not reply_json.get("need_more_context"):
                log_ai_interaction(logger, f"gen_simple_reply (attempt {attempt+1})", system_prompt, user_prompt, reply, "SUCCESS")
                break

            log_ai_interaction(logger, f"gen_simple_reply (attempt {attempt+1})", system_prompt, user_prompt, reply, "INFO", "Need more context, retrying...")
            max_msg += 10

        except Exception as e:
            log_ai_interaction(logger, f"gen_simple_reply (attempt {attempt+1})", system_prompt, user_prompt, reply, "FAILURE", e)
            return ""

    if reply_json.get("should_reply"):
        final_reply = reply_json.get("reply", "")

        if memory and final_reply:
            try:
                memory.remember_reply(final_reply)
                logger.info("Memory updated with simple reply.")
            except Exception as e:
                logger.error(f"Failed to update memory with simple reply: {e}")

        return final_reply
    else:
        return ""
