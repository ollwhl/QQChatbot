from typing import Optional, List
from datetime import datetime
import os


class PersonaMemory:
    def __init__(self, group_id: int):
        self.group_id = group_id

        # === 最近回复记录（避免重复） ===
        self.recent_replies: List[str] = []
        self.recent_replies_time: List[str] = []
        self.last_spoken_time: datetime | None = None

        # === 立场 / 承诺（弱约束）===
        # e.g. {"arknights": "玩过", "genshin": "最近没怎么玩"}
        self.positions: dict[str, str] = {}

        # === 最近参与的话题（用于避免插嘴/复读）===
        self.recent_topics: List[str] = []
        self.recent_topics_time: List[str] = []
        self.is_blank = True

        # === 记忆日志文件路径 ===
        self._log_dir = "./memory_logs"
        os.makedirs(self._log_dir, exist_ok=True)
        self._log_file = os.path.join(self._log_dir, f"{group_id}.log")

    def _write_log(self, log_type: str, content: str):
        """写入记忆日志到文件"""
        try:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            log_line = f"[{timestamp}] [{log_type}] {content}\n"
            with open(self._log_file, "a", encoding="utf-8") as f:
                f.write(log_line)
        except Exception as e:
            print(f"写入记忆日志失败: {e}")

    def remember_reply(self, reply_text: str | None, max_len: int = 5):
        """在确认回复被发送后调用"""
        if not reply_text:
            return
        self.recent_replies.append(reply_text)
        self.recent_replies_time.append(datetime.now().strftime("%H:%M:%S"))
        if len(self.recent_replies) > max_len:
            self.recent_replies.pop(0)
            self.recent_replies_time.pop(0)
        self.last_spoken_time = datetime.now()
        self.is_blank = False

        # 写入日志
        self._write_log("AI回复", reply_text)

    def remember_position(self, key: str, value: str):
        self.positions[key] = value
        self.is_blank = False

    def remember_topic(self, topic: str, max_len: int = 5):
        if not topic:
            return
        self.recent_topics.append(topic)
        self.recent_topics_time.append(datetime.now().strftime("%Y:%m:%d:%H:%M:%S"))
        if len(self.recent_topics) > max_len:
            self.recent_topics.pop(0)
            self.recent_topics_time.pop(0)
        self.is_blank = False

        # 写入日志
        self._write_log("新话题", topic)

    def to_str(self):
        parts = []

        # 最近回复
        if self.recent_replies:
            replies = []
            for i in range(len(self.recent_replies)):
                replies.append(f"[{self.recent_replies_time[i]}] {self.recent_replies[i]}")
            parts.append("你最近说过的话（避免重复）：\n" + "\n".join(replies))

        # 立场
        if self.positions:
            parts.append(f"你表达过的立场：{str(self.positions)}")

        # 话题
        if self.recent_topics:
            topics = []
            for i in range(len(self.recent_topics)):
                topics.append(f"[{self.recent_topics_time[i]}] {self.recent_topics[i]}")
            parts.append("你参与过的话题：\n" + "\n".join(topics))

        return "\n".join(parts)
