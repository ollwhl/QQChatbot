# user_relationships.py
from datetime import datetime
from typing import Dict, List, Optional, Any
import json
import os
import qq_msg
class Userrelationship:
    """用户基础档案类"""
    
    def __init__(self, user_id: int):
        self.user_id = user_id
        
        # 基础信息
        self.basic_info = {
            "known_name": "",           # 已知的名字/昵称
            "preferred_name": "",       # 偏好的称呼
            "gender": "unknown",        # 性别
            "age_group": "unknown",     # 年龄组
            "first_interaction": None,  # 第一次互动时间
            "last_interaction": None,   # 最后一次互动时间
        }
        
        # 对话特征
        self.conversation_style = {
            "formality_level": 0.5,     # 正式程度 (0-1)
            "response_speed": "normal", # 回复速度 (fast/normal/slow)
            "message_length": "medium", # 消息长度 (short/medium/long)
            "emoticon_usage": "medium", # 表情使用频率 (low/medium/high)
            "language_style": "neutral",# 语言风格 (casual/formal/neutral)
        }
        
        # 兴趣标签
        self.interests = {
            "confirmed_interests": [],   # 确认的兴趣
            "possible_interests": [],    # 可能的兴趣
            "mentioned_topics": [],      # 提到过的话题
        }
        
        # 关系状态
        self.relationship = {
            "interaction_count": 0,      # 互动次数
            "familiarity_level": 0.0,    # 熟悉程度 (0-1)
            "trust_level": 0.0,          # 信任程度 (0-1)
            "relationship_stage": "stranger",  # 关系阶段
        }
        
        # 记忆亮点
        self.memory_highlights = {
            "significant_conversations": [],  # 重要对话
            "shared_jokes": [],               # 共享的笑话
            "personal_stories": [],           # 个人故事
        }
        
        # 情感模式
        self.emotional_patterns = {
            "common_moods": [],               # 常见情绪
            "comfort_topics": [],             # 安慰话题
            "excitement_triggers": [],        # 兴奋触发点
        }
        
        # 统计信息
        self.statistics = {
            "total_messages": 0,              # 总消息数
            "avg_response_time": 0,           # 平均回复时间
            "active_days": 0,                 # 活跃天数
            "conversation_depth": 0.0,        # 对话深度评分
        }
    
    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            "user_id": self.user_id,
            "basic_info": self.basic_info,
            "conversation_style": self.conversation_style,
            "interests": self.interests,
            "relationship": self.relationship,
            "memory_highlights": self.memory_highlights,
            "emotional_patterns": self.emotional_patterns,
            "statistics": self.statistics,
        }


# ──────────────── LLM 驱动的用户画像生成 ────────────────

# relationship 生成的系统提示词
relationship_GENERATION_PROMPT = """你是一个专业的人际关系分析师。基于用户的聊天记录，分析该用户与目标对象（我）的关系。

请从以下维度分析：
1. **关系亲密度**：根据聊天频率、互动深度、话题私密性等评估关系亲密程度（陌生/普通/熟悉/亲密/密友）
2. **互动模式**：该用户与我的互动方式（主动/被动、热情/冷淡、正式/随意等）
3. **共同话题**：我们经常聊什么，有哪些共同兴趣
4. **情感倾向**：该用户对我的态度（友好/中立/疏远/崇拜/依赖等）
5. **关系特点**：这段关系的独特之处，特殊的互动模式或记忆点
6. **对我的称呼**：该用户通常如何称呼我（昵称、正式称呼等）
7. **互动频率**：该用户与我的互动活跃度（高/中/低）
8. **关系建议**：基于当前关系状态，给出维护或改善关系的建议

**重要要求**：
- 基于客观聊天记录分析，避免过度推测
- 重点关注该用户与"我"（目标对象）的双向互动，而非单纯描述该用户的个人特征
- 如果某些维度信息不足，标注为"信息不足，待观察"
- 保持中立和尊重的态度
- 输出格式必须是合法的 JSON

输出 JSON 格式：
{
  "relationship_closeness": "关系亲密度（陌生/普通/熟悉/亲密/密友）",
  "interaction_pattern": "互动模式描述",
  "common_topics": ["共同话题1", "共同话题2", ...],
  "emotional_tendency": "情感倾向描述",
  "relationship_features": "关系特点描述",
  "how_they_call_me": "对我的称呼",
  "interaction_frequency": "互动频率（高/中/低）",
  "relationship_advice": "关系建议",
  "tags": ["关系标签1", "关系标签2", "关系标签3"]
}
"""


def generate_user_relationship_llm(
    user_id: int,
    target_user_id: int,
    group_id: Optional[int] = None,
    message_limit: int = 100,
    force_update: bool = False
) -> Dict[str, Any]:
    """
    使用 LLM 分析用户与目标对象的关系，生成关系画像并保存到文件。

    Args:
        user_id: 被分析的用户 ID
        target_user_id: 关系分析的目标对象 ID（通常是机器人/主人的ID）
        group_id: 群组 ID（可选，如果指定则只分析该群的消息）
        message_limit: 分析的消息数量上限
        force_update: 是否强制更新（如果为 False 且文件已存在，则跳过生成）

    Returns:
        Dict: 生成的关系画像 JSON
    """
    from database import db
    import call_llm

    # 确保 relationship 目录存在
    relationship_dir = "./relationship"
    os.makedirs(relationship_dir, exist_ok=True)

    relationship_path = os.path.join(relationship_dir, f"{user_id}.json")

    # 如果文件已存在且不强制更新，直接读取返回
    if not force_update and os.path.exists(relationship_path):
        print(f"[relationship] 用户 {user_id} 的画像已存在，跳过生成")
        with open(relationship_path, "r", encoding="utf-8") as f:
            return json.load(f)

    print(f"[relationship] 开始生成用户 {user_id} 的画像...")

    # 获取群组的历史消息（而非单个用户的消息，这样才能看到互动）
    if group_id:
        # 获取该群组的消息，这样可以看到该用户与目标对象的互动
        messages = db.get_latest_messages_by_time(group_id, True, 60 * 24 * 7)  # 最近7天
        # 限制消息数量
        if len(messages) > message_limit:
            messages = messages[-message_limit:]
    else:
        # 私聊场景：获取该用户与目标对象的私聊消息
        messages = db.get_latest_messages_by_time(user_id, False, 60 * 24 * 7)  # 最近7天
        if len(messages) > message_limit:
            messages = messages[-message_limit:]

    if not messages:
        print(f"[relationship] 用户 {user_id} 没有足够的消息记录，无法生成画像")
        return {
            "error": "消息记录不足",
            "user_id": user_id,
            "target_user_id": target_user_id,
            "message_count": 0
        }

    # 从消息中获取目标用户的昵称
    target_user_name = None
    for msg in messages:
        if msg.user_id == target_user_id:
            if msg.sender_card:
                target_user_name = f"{msg.sender_card}({msg.sender_name})"
            else:
                target_user_name = msg.sender_name
            break

    # 如果没找到目标用户的消息，使用默认值
    if not target_user_name:
        from config import CONFIG
        target_user_name = CONFIG.get("master_name", "目标用户")

    # 构建消息文本（区分该用户和目标对象的消息）
    # 使用格式：群昵称(QQ昵称) 或 QQ昵称
    message_texts = []
    for msg in messages:
        # 格式：[时间] 发送者: 消息内容
        time_str = msg.timestamp
        formatted_time = f"{time_str[0:4]}-{time_str[4:6]}-{time_str[6:8]} {time_str[8:10]}:{time_str[10:12]}"

        # 构建发送者显示名称
        if msg.sender_card and msg.sender_name:
            display_name = f"{msg.sender_card}({msg.sender_name})"
        else:
            display_name = msg.sender_name or msg.sender_card or "未知用户"

        # 标记发送者
        if msg.user_id == target_user_id:
            sender = f"【我: {display_name}】"
        elif msg.user_id == user_id:
            sender = f"【该用户: {display_name}】"
        else:
            sender = display_name

        message_texts.append(f"[{formatted_time}] {sender}: {msg.context}")

    chat_history = "\n".join(message_texts)

    # 使用 LLM 生成关系画像
    print(f"[relationship] 分析 {len(messages)} 条消息...")
    user_prompt = f"""分析以下聊天记录，生成【该用户】与【我】的关系画像：

**分析对象**：
- 该用户ID: {user_id}
- 我的ID: {target_user_id}
- 我的昵称: {target_user_name}

**分析范围**：
- 消息数量: {len(messages)} 条
{'- 群组ID: ' + str(group_id) if group_id else '- 场景: 私聊'}

**聊天记录**：
{chat_history}

**分析任务**：
请重点分析【该用户】与【我】之间的关系，包括互动模式、亲密度、情感倾向等。
聊天记录中标记为【我】的消息是目标对象发送的，标记为【该用户】的消息是被分析者发送的。
其他未标记的消息是群组中其他成员发送的，可作为背景参考。
注意：昵称格式为"群昵称(QQ昵称)"或"QQ昵称"，分析时请注意同一个人可能以不同昵称被提及。

请按照要求的 JSON 格式输出关系画像。"""

    try:
        # 调用 LLM 生成画像
        relationship_text = call_llm.call_chat_complete(
            system_prompt=relationship_GENERATION_PROMPT,
            user_prompt=user_prompt,
            max_tokens=1500,
            temperature=0.3,  # 较低温度以保持分析的一致性
            timeout=90,
        )

        # 解析 JSON
        # 尝试提取 JSON（可能 LLM 会返回带 markdown 代码块的内容）
        relationship_text = relationship_text.strip()
        if relationship_text.startswith("```json"):
            relationship_text = relationship_text[7:]
        if relationship_text.startswith("```"):
            relationship_text = relationship_text[3:]
        if relationship_text.endswith("```"):
            relationship_text = relationship_text[:-3]
        relationship_text = relationship_text.strip()

        relationship_data = json.loads(relationship_text)

        # 补充元数据
        relationship_data["user_id"] = user_id
        extra_data = json.loads(qq_msg.get_stranger_info(user_id)).get("data")
        relationship_data["user_name"] = extra_data.get("nick")
        relationship_data["user_sex"] = extra_data.get("sex")
        relationship_data["user_age"] = extra_data.get("age")
        relationship_data["target_user_id"] = target_user_id
        relationship_data["target_user_name"] = target_user_name
        if group_id:
            relationship_data["group_id"] = group_id
        relationship_data["message_count"] = len(messages)
        relationship_data["last_updated"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # 保存到文件
        with open(relationship_path, "w", encoding="utf-8") as f:
            json.dump(relationship_data, f, ensure_ascii=False, indent=2)

        print(f"[relationship] 用户 {user_id} 的画像已生成并保存到 {relationship_path}")
        return relationship_data

    except json.JSONDecodeError as e:
        print(f"[relationship] JSON 解析失败: {e}")
        print(f"[relationship] LLM 返回内容: {relationship_text[:500]}")
        # 保存原始文本供调试
        error_data = {
            "error": "JSON解析失败",
            "user_id": user_id,
            "target_user_id": target_user_id,
            "raw_response": relationship_text,
            "message_count": len(messages)
        }
        with open(relationship_path, "w", encoding="utf-8") as f:
            json.dump(error_data, f, ensure_ascii=False, indent=2)
        return error_data

    except Exception as e:
        print(f"[relationship] 生成画像失败: {e}")
        return {
            "error": str(e),
            "user_id": user_id,
            "target_user_id": target_user_id,
            "message_count": len(messages)
        }


def load_user_relationship(user_id: int) -> Optional[Dict[str, Any]]:
    """
    加载用户画像。

    Args:
        user_id: 用户 ID

    Returns:
        Dict or None: 用户画像数据，如果不存在则返回 None
    """
    relationship_path = f"./relationship/{user_id}.json"

    if not os.path.exists(relationship_path):
        return None

    try:
        with open(relationship_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"[relationship] 加载用户 {user_id} 画像失败: {e}")
        return None


def update_user_relationship(
    user_id: int,
    target_user_id: int,
    group_id: Optional[int] = None,
    message_limit: int = 100
) -> Dict[str, Any]:
    """
    更新用户关系画像（强制重新生成）。

    Args:
        user_id: 用户 ID
        target_user_id: 关系分析的目标对象 ID
        group_id: 群组 ID（可选）
        message_limit: 分析的消息数量上限

    Returns:
        Dict: 更新后的关系画像
    """
    return generate_user_relationship_llm(user_id, target_user_id, group_id, message_limit, force_update=True)


def batch_generate_relationships(
    group_id: int,
    target_user_id: int,
    min_message_count: int = 20,
    limit: Optional[int] = None
):
    """
    批量生成群组内所有活跃用户与目标对象的关系画像。

    Args:
        group_id: 群组 ID
        target_user_id: 关系分析的目标对象 ID
        min_message_count: 最少消息数量（少于此数量的用户不生成画像）
        limit: 限制生成的用户数量
    """
    from database import db

    # 获取群组的所有消息
    messages = db.get_latest_messages_by_count(group_id, True, 1000)

    # 统计每个用户的消息数量
    user_message_counts = {}
    for msg in messages:
        if not msg.is_ai and msg.user_id != target_user_id:  # 排除 AI 和目标用户自己
            user_id = msg.user_id
            user_message_counts[user_id] = user_message_counts.get(user_id, 0) + 1

    # 过滤出活跃用户
    active_users = [
        user_id for user_id, count in user_message_counts.items()
        if count >= min_message_count
    ]

    # 按消息数量排序
    active_users.sort(key=lambda uid: user_message_counts[uid], reverse=True)

    if limit:
        active_users = active_users[:limit]

    print(f"[relationship] 群组 {group_id} 中有 {len(active_users)} 个活跃用户待生成关系画像")

    # 批量生成
    results = {}
    for i, user_id in enumerate(active_users, 1):
        print(f"[relationship] 进度: {i}/{len(active_users)}")
        relationship = generate_user_relationship_llm(user_id, target_user_id, group_id)
        results[user_id] = relationship

    print(f"[relationship] 批量生成完成，共生成 {len(results)} 个关系画像")
    return results


if __name__ == "__main__":
    # 测试示例
    import sys
    from config import CONFIG

    # 从配置读取目标用户（通常是机器人主人）
    default_target_user_id = CONFIG.get("master_user_id", 0)

    if len(sys.argv) < 2:
        print("用法: python user_relationship.py <user_id> [target_user_id] [group_id]")
        print("或: python user_relationship.py batch <group_id> [target_user_id]")
        print(f"\n默认目标用户ID: {default_target_user_id}")
        print("注意：目标用户昵称会从聊天记录中自动获取")
        sys.exit(1)

    if sys.argv[1] == "batch":
        # 批量生成
        group_id = int(sys.argv[2])
        target_user_id = int(sys.argv[3]) if len(sys.argv) > 3 else default_target_user_id

        batch_generate_relationships(group_id, target_user_id)
    else:
        # 单个用户
        user_id = int(sys.argv[1])
        target_user_id = int(sys.argv[2]) if len(sys.argv) > 2 else default_target_user_id
        group_id = int(sys.argv[3]) if len(sys.argv) > 3 else None

        relationship = generate_user_relationship_llm(user_id, target_user_id, group_id, force_update=True)
        print("\n生成的关系画像:")
        print(json.dumps(relationship, ensure_ascii=False, indent=2))