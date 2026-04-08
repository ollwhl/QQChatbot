# user_profiles.py
from datetime import datetime
from typing import Dict, List, Optional, Any
import json
import os
import qq_msg
class UserProfile:
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

# Profile 生成的系统提示词
PROFILE_GENERATION_PROMPT = """你是一个专业的用户画像分析师。基于用户的聊天记录，生成一份详细的用户画像。

请从以下维度分析用户：
1. **性格特征**：从聊天内容推测用户的性格倾向（如外向/内向、理性/感性、幽默/严肃等）
2. **兴趣爱好**：用户经常谈论的话题、感兴趣的领域
3. **沟通风格**：用户的表达习惯、常用语气、互动方式
4. **活跃时间**：用户通常在什么时间段发言
5. **关键标签**：用3-5个关键词总结用户特征
6. **其他特征**：任何值得记录的特殊行为模式或习惯

**重要要求**：
- 基于客观事实进行分析，避免过度推测
- 如果某些维度信息不足，标注为"信息不足，待补充"
- 保持中立和尊重的态度
- 输出格式必须是合法的 JSON

输出 JSON 格式：
{
  "personality": "性格特征描述",
  "interests": ["兴趣1", "兴趣2", ...],
  "communication_style": "沟通风格描述",
  "active_time": "活跃时间描述",
  "tags": ["标签1", "标签2", "标签3"],
  "other_notes": "其他备注",
  "message_count": 消息数量,
  "last_updated": "更新时间"
}
"""


def generate_user_profile_llm(
    user_id: int,
    group_id: Optional[int] = None,
    message_limit: int = 300,
    force_update: bool = False
) -> Dict[str, Any]:
    """
    使用 LLM 生成用户画像并保存到文件。

    Args:
        user_id: 用户 ID
        group_id: 群组 ID（可选，如果指定则只分析该群的消息）
        message_limit: 分析的消息数量上限
        force_update: 是否强制更新（如果为 False 且文件已存在，则跳过生成）

    Returns:
        Dict: 生成的用户画像 JSON
    """
    from database import db
    import call_llm

    # 确保 profile 目录存在
    profile_dir = "./profile"
    os.makedirs(profile_dir, exist_ok=True)

    profile_path = os.path.join(profile_dir, f"{user_id}.json")

    # 如果文件已存在且不强制更新，直接读取返回
    if not force_update and os.path.exists(profile_path):
        print(f"[Profile] 用户 {user_id} 的画像已存在，跳过生成")
        with open(profile_path, "r", encoding="utf-8") as f:
            return json.load(f)

    print(f"[Profile] 开始生成用户 {user_id} 的画像...")

    # 获取用户的历史消息
    if group_id:
        # 获取该用户在指定群组的消息
        messages = db.get_latest_messages_by_sender_id(user_id, limit=message_limit)
        # 过滤出该群的消息
        messages = [msg for msg in messages if msg.group_id == group_id and msg.is_group]
    else:
        # 获取用户的所有消息
        messages = db.get_latest_messages_by_sender_id(user_id, limit=message_limit)

    if not messages:
        print(f"[Profile] 用户 {user_id} 没有足够的消息记录，无法生成画像")
        return {
            "error": "消息记录不足",
            "user_id": user_id,
            "message_count": 0
        }

    # 构建消息文本（包含时间信息）
    message_texts = []
    for msg in messages:
        # 格式：[时间] 消息内容
        time_str = msg.timestamp
        formatted_time = f"{time_str[0:4]}-{time_str[4:6]}-{time_str[6:8]} {time_str[8:10]}:{time_str[10:12]}"
        message_texts.append(f"[{formatted_time}] {msg.context}")

    chat_history = "\n".join(message_texts)

    # 使用 LLM 生成画像
    print(f"[Profile] 分析 {len(messages)} 条消息...")
    user_prompt = f"""分析以下用户的聊天记录，生成用户画像：

用户ID: {user_id}
消息数量: {len(messages)}
{'群组ID: ' + str(group_id) if group_id else ''}

聊天记录：
{chat_history}

请按照要求的 JSON 格式输出用户画像。"""

    try:
        # 调用 LLM 生成画像
        profile_text = call_llm.call_chat_complete(
            system_prompt=PROFILE_GENERATION_PROMPT,
            user_prompt=user_prompt,
            max_tokens=1500,
            temperature=0.3,  # 较低温度以保持分析的一致性
            timeout=90,
        )

        # 解析 JSON
        # 尝试提取 JSON（可能 LLM 会返回带 markdown 代码块的内容）
        profile_text = profile_text.strip()
        if profile_text.startswith("```json"):
            profile_text = profile_text[7:]
        if profile_text.startswith("```"):
            profile_text = profile_text[3:]
        if profile_text.endswith("```"):
            profile_text = profile_text[:-3]
        profile_text = profile_text.strip()

        profile_data = json.loads(profile_text)

        # 补充元数据
        profile_data["user_id"] = user_id
        extra_data = json.loads(qq_msg.get_stranger_info(user_id)).get("data")
        profile_data["user_name"] = extra_data.get("nick")
        profile_data["user_sex"] = extra_data.get("sex")
        profile_data["user_age"] = extra_data.get("age")
        if group_id:
            profile_data["group_id"] = group_id
        profile_data["message_count"] = len(messages)
        profile_data["last_updated"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # 保存到文件
        with open(profile_path, "w", encoding="utf-8") as f:
            json.dump(profile_data, f, ensure_ascii=False, indent=2)

        print(f"[Profile] 用户 {user_id} 的画像已生成并保存到 {profile_path}")
        return profile_data

    except json.JSONDecodeError as e:
        print(f"[Profile] JSON 解析失败: {e}")
        print(f"[Profile] LLM 返回内容: {profile_text[:500]}")
        # 保存原始文本供调试
        error_data = {
            "error": "JSON解析失败",
            "user_id": user_id,
            "raw_response": profile_text,
            "message_count": len(messages)
        }
        with open(profile_path, "w", encoding="utf-8") as f:
            json.dump(error_data, f, ensure_ascii=False, indent=2)
        return error_data

    except Exception as e:
        print(f"[Profile] 生成画像失败: {e}")
        return {
            "error": str(e),
            "user_id": user_id,
            "message_count": len(messages)
        }


def load_user_profile(user_id: int) -> Optional[Dict[str, Any]]:
    """
    加载用户画像。

    Args:
        user_id: 用户 ID

    Returns:
        Dict or None: 用户画像数据，如果不存在则返回 None
    """
    profile_path = f"./profile/{user_id}.json"

    if not os.path.exists(profile_path):
        return None

    try:
        with open(profile_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"[Profile] 加载用户 {user_id} 画像失败: {e}")
        return None


def update_user_profile(user_id: int, group_id: Optional[int] = None, message_limit: int = 100) -> Dict[str, Any]:
    """
    更新用户画像（强制重新生成）。

    Args:
        user_id: 用户 ID
        group_id: 群组 ID（可选）
        message_limit: 分析的消息数量上限

    Returns:
        Dict: 更新后的用户画像
    """
    return generate_user_profile_llm(user_id, group_id, message_limit, force_update=True)


def batch_generate_profiles(group_id: int, min_message_count: int = 20, limit: Optional[int] = None):
    """
    批量生成群组内所有活跃用户的画像。

    Args:
        group_id: 群组 ID
        min_message_count: 最少消息数量（少于此数量的用户不生成画像）
        limit: 限制生成的用户数量
    """
    from database import db

    # 获取群组的所有消息
    messages = db.get_latest_messages_by_count(group_id, True, 1000)

    # 统计每个用户的消息数量
    user_message_counts = {}
    for msg in messages:
        if not msg.is_ai:  # 排除 AI 自己的消息
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

    print(f"[Profile] 群组 {group_id} 中有 {len(active_users)} 个活跃用户待生成画像")

    # 批量生成
    results = {}
    for i, user_id in enumerate(active_users, 1):
        print(f"[Profile] 进度: {i}/{len(active_users)}")
        profile = generate_user_profile_llm(user_id, group_id)
        results[user_id] = profile

    print(f"[Profile] 批量生成完成，共生成 {len(results)} 个用户画像")
    return results


if __name__ == "__main__":
    # 测试示例
    import sys

    if len(sys.argv) < 2:
        print("用法: python user_profile.py <user_id> [group_id]")
        print("或: python user_profile.py batch <group_id>")
        sys.exit(1)

    if sys.argv[1] == "batch":
        # 批量生成
        group_id = int(sys.argv[2])
        batch_generate_profiles(group_id)
    else:
        # 单个用户
        user_id = int(sys.argv[1])
        group_id = int(sys.argv[2]) if len(sys.argv) > 2 else None

        profile = generate_user_profile_llm(user_id, group_id, force_update=True)
        print("\n生成的用户画像:")
        print(json.dumps(profile, ensure_ascii=False, indent=2))