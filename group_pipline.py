from datetime import datetime
from typing import List, Dict, Optional
from call_llm import call_chat_complete, CHAT_MODEL, MESSAGE_ANALYZE_MODEL, IMAGE_MODEL
import json
import re
from functools import lru_cache
from config import CONFIG
from logger import group_chat_logger, private_chat_logger, log_ai_interaction
from database import db , MessageModel
from memory import PersonaMemory


"""
极度社恐再三确认的回复模式
"""


class ConversationTurn:
    def __init__(self, msgs: List[MessageModel]):
        # === 原始事实 ===
        self.trigger_msg: MessageModel = msgs[-1]
        self.history_msg: List[MessageModel] = msgs[:-1]
        self.all_msgs: List[MessageModel] = msgs

        # === 语境判断层 ===
        self.is_addressed_to_me: bool = False      # 是否被点名 / @ / 明显指向
        self.is_reply_context: bool = False        # 是否是接 AI 上一句
        self.social_role: str = "observer"
        # observer | participant | expert | joker | moderator

         # ===== 强制回复判断 =====
        self.reply_obligated: bool = False

        # === 对话理解层 ===
        self.topic_summary: str | None = None
        self.current_topic_stage: str | None = None
        # opening | ongoing | closing | fragmented
        self.vibe: str | None = None

        self.intent_type: str = "unknown"
        # question | discussion | emotion | info_dump | noise

        # === 决策层 ===
        self.should_reply: bool = False
        self.reply_priority: float = 0.0   # 0~1
        self.reply_style: str | None = None
        # short | normal | playful | technical | silent

        # === 生成层 ===
        self.final_reply: str | None = None
        
        #=== 人格记忆 ===
        self.memory:PersonaMemory


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

def simple_group_chat(
    logger,
    msg_models: list[MessageModel],
    memory: PersonaMemory
):
    # ===== 0. 防御式检查 =====
    if not msg_models:
        return None, memory
    turn = ConversationTurn(msg_models)
    turn.memory = memory

# 入口
def handle_group_chat(
    logger,
    msg_models: list[MessageModel],
    memory: PersonaMemory
):
    # ===== 0. 防御式检查 =====
    if not msg_models:
        return None, memory
    
    # ===== 1. 构建 turn =====
    turn = ConversationTurn(msg_models)
    turn.memory = memory
    # ===== 2. 社会语境分析 =====
    turn = analyze_social_context(logger, turn)

    # ===== 3. 强制回复直通 =====
    if turn.reply_obligated:
        turn.should_reply = True
    else:
        # ===== 4. 对话理解（必要时）=====
        if len(turn.all_msgs) >= 3:
            turn = analyze_topic_and_intent(logger, turn)
        else:
            # 明确标记为碎片化
            turn.current_topic_stage = "fragmented"
            turn.intent_type = "statement"

        # ===== 5. 决定是否参与 =====
        turn = decide_participation(logger, turn)

    # ===== 6. 最终不回复 =====
    if not turn.should_reply:
        return "", memory

    # ===== 7. 回复风格规划 =====
    turn = plan_reply_style(logger, turn)

    # ===== 8. 生成回复 =====
    turn = generate_group_reply(logger, turn)

    if not turn or  "No reply" in str(turn.final_reply):
        return "", memory

    # ===== 9. 提交人格记忆（只在确认回复后）=====
    memory.remember_reply(turn.final_reply)
    if turn.topic_summary:
        memory.remember_topic(turn.topic_summary)

    return turn.final_reply, memory




def check_reply_obligation(turn: ConversationTurn) -> bool:
    if turn.is_addressed_to_me:
        return True
    if turn.is_reply_context:
        return True
    return False


# Prompt 缓存 - 避免每次请求都读取文件
#@lru_cache(maxsize=10)
def load_prompt(filename: str) -> str:
    """加载并缓存 prompt 文件"""
    filepath = "./prompts/group/"+filename
    with open(filepath, "r", encoding='utf-8') as f:
        return f.read().replace('$master$', CONFIG['master_name'])
    
#-------------------------analyze Pipline----------------------#

def analyze_social_context(logger, turn: ConversationTurn, max_msg=15):
    """
    社会语境分析

    "is_addressed_to_me" 判断消息是否是在对AI说话

    "is_reply_context" 是不是回复型的消息

    "social_role": 社会角色定义
        - observer：旁观者，不在当前对话回合
        - participant：对话参与者，但未被直接点名
        - expert：被隐含期待提供信息或判断
        - joker：当前语境适合轻松、玩笑式插话
        - moderator：被期待维持话题或秩序（极少）

    """
    user_prompt = "\n".join(
        m.to_str() for m in turn.all_msgs[-max_msg:]
    )

    system_prompt = load_prompt(
        "group_social_context_prompt.txt"
    )

    reply = None
    try:
        reply = call_chat_complete(
            system_prompt,
            user_prompt,
            1500,
            0.0,
            120
        )
        data = json.loads(extract_json(reply))

        turn.is_addressed_to_me = data["is_addressed_to_me"]
        turn.is_reply_context = data["is_reply_context"]
        turn.social_role = data["social_role"]

        log_ai_interaction(
            logger, "analyze_social_context",
            system_prompt, user_prompt, reply, "SUCCESS"
        )
    except Exception as e:
        log_ai_interaction(
            logger, "analyze_social_context",
            system_prompt, user_prompt,
            reply or "(no reply)", "FAILURE", e
        )

    return turn


def analyze_topic_and_intent(logger, turn: ConversationTurn, max_msg=20):
    """
    话题与意图分析

    "topic_summary" 总结聊天内容

    "current_topic_stage" 话题阶段定义
            - opening：刚引入的话题
            - ongoing：多人正在围绕讨论
            - closing：话题明显接近结束
            - fragmented：话题混乱或已经分散

    "intent_type" 意图类型定义
            - question：提问、寻求答案
            - discussion：观点交流、讨论
            - emotion：情绪表达、感叹、吐槽
            - info_dump：单向信息投放
            - noise：表情、无意义、刷屏

    """
    user_prompt = "\n".join(
        m.to_str() for m in turn.all_msgs[-max_msg:]
    )

    system_prompt = load_prompt(
        "group_topic_intent_prompt.txt"
    )

    reply = None
    try:
        reply = call_chat_complete(
            system_prompt,
            user_prompt,
            2000,
            0.0,
            180
        )
        data = json.loads(extract_json(reply))

        turn.topic_summary = data["topic_summary"]
        turn.current_topic_stage = data["current_topic_stage"]
        turn.intent_type = data["intent_type"]

        log_ai_interaction(
            logger, "analyze_topic_and_intent",
            system_prompt, user_prompt, reply, "SUCCESS"
        )
    except Exception as e:
        log_ai_interaction(
            logger, "analyze_topic_and_intent",
            system_prompt, user_prompt,
            reply or "(no reply)", "FAILURE", e
        )

    return turn



def decide_participation(logger, turn: ConversationTurn):
    """
    参与决策层

    "should_reply" 是否应该回复
    "reply_priority": 回复必要性强度（0~1）
    """

    system_prompt = load_prompt(
        "group_participation_decision_prompt.txt"
    )

    user_prompt = json.dumps({
        "is_addressed_to_me": turn.is_addressed_to_me,
        "is_reply_context": turn.is_reply_context,
        "social_role": turn.social_role,
        "topic_summary": turn.topic_summary,
        "topic_stage": turn.current_topic_stage,
        "intent_type": turn.intent_type,
        "memory": turn.memory.to_str(),
        "messages":build_group_user_prompt(
        messages=[m.to_str() for m in turn.all_msgs],
        context_info=f"""
话题总结：{turn.topic_summary}
你的群聊角色：{turn.social_role}
回复风格：{turn.reply_style}
""".strip(),
        memory_str = "",
        max_messages=5
    )
        
    }, ensure_ascii=False, indent=2)

    reply = None
    try:
        reply = call_chat_complete(
            system_prompt,
            user_prompt,
            800,
            0.0,
            60
        )
        data = json.loads(extract_json(reply))

        turn.should_reply = data["should_reply"]
        turn.reply_priority = data["reply_priority"]

        log_ai_interaction(
            logger, "decide_participation",
            system_prompt, user_prompt, reply, "SUCCESS"
        )
    except Exception as e:
        log_ai_interaction(
            logger, "decide_participation",
            system_prompt, user_prompt,
            reply or "(no reply)", "FAILURE", e
        )

    return turn


def plan_reply_style(logger, turn: ConversationTurn):
    """
    回复规划层

    "reply_style" 回复风格定义
        - short：一句话、极简回应
        - normal：自然日常回复
        - playful：轻松、玩笑、不严肃
        - technical：偏理性、解释型
        - silent：即使 should_reply 为 true，也选择克制表达

    """
    if not turn.should_reply:
        return turn

    system_prompt = load_prompt(
        "group_reply_planning_prompt.txt"
    )

    user_prompt = json.dumps({
        "topic_summary": turn.topic_summary,
        "intent_type": turn.intent_type,
        "social_role": turn.social_role,
        "priority": turn.reply_priority
    }, ensure_ascii=False, indent=2)

    reply = None
    try:
        reply = call_chat_complete(
            system_prompt,
            user_prompt,
            500,
            0.3,
            60
        )
        data = json.loads(extract_json(reply))
        turn.reply_style = data["reply_style"]

        log_ai_interaction(
            logger, "plan_reply_style",
            system_prompt, user_prompt, reply, "SUCCESS"
        )
    except Exception as e:
        log_ai_interaction(
            logger, "plan_reply_style",
            system_prompt, user_prompt,
            reply or "(no reply)", "FAILURE", e
        )

    return turn


def generate_group_reply(logger, turn: ConversationTurn, max_msg=20):
    """
    群聊回复生成层

    实际回复的内容

    """
    if not turn.should_reply:
        return None

    system_prompt = load_prompt(
        "group_chat_response_prompt.txt"
    )

    user_prompt = build_group_user_prompt(
        messages=[m.to_str() for m in turn.all_msgs],
        context_info=f"""
话题总结：{turn.topic_summary}
你的群聊角色：{turn.social_role}
回复风格：{turn.reply_style}
""".strip(),
        memory_str = turn.memory.to_str() if not turn.memory.is_blank else "",
        max_messages=max_msg
    )

    reply = None
    try:
        reply = call_chat_complete(
            system_prompt,
            user_prompt,
            1000,
            1.2,
            60
        )
        turn.final_reply = reply

        log_ai_interaction(
            logger, "generate_group_reply",
            system_prompt, user_prompt, reply, "SUCCESS"
        )
        return turn
    except Exception as e:
        log_ai_interaction(
            logger, "generate_group_reply",
            system_prompt, user_prompt,
            reply or "(no reply)", "FAILURE", e
        )
        return None
#--------------------simple pipline-------------------- 
def build_group_user_prompt(
    messages: List[str],
    context_info: Optional[str] = None,
    memory_str = None,
    max_messages: int = 20
) -> str:
    """
    为对话型AI机器人构建user prompt
    
    参数:
        messages: 消息列表，每条消息格式为 {"sender": "发送者昵称", "content": "消息内容", "time": "时间(可选)"}
        context_info: 背景信息
        max_messages: 最多包含的历史消息条数
        
    返回:
        构建好的user prompt字符串
    """
    
    # 构建prompt各部分
    prompt_parts = []
    
    # 2. 额外上下文
    if context_info:
        prompt_parts.append(f"\n【背景信息】\n{context_info}")
    
    if memory_str != "":
        prompt_parts.append(f"\n【历史信息】\n{memory_str}")

    
    # 3. 消息历史
    prompt_parts.append("\n【最近的群聊消息】")
    
    # 限制消息数量，取最新的几条
    recent_messages = messages[-max_messages:] if len(messages) > max_messages else messages
    if not recent_messages:
        prompt_parts.append("(暂无消息记录)")
    else:
        for msg in recent_messages:
            prompt_parts.append(msg)
    
    return "\n".join(prompt_parts)