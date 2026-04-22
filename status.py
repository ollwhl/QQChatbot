import threading
import json
import os
from collections import deque
from typing import Optional
from memory import PersonaMemory
from config import CONFIG

_THRESHOLDS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chat_thresholds.json")
_MAX_RECENT_MSG_IDS = 100

# 从 config 读取默认阈值
_threshold_cfg = CONFIG.get("chatbot_server", {}).get("reply_threshold", {})
_DEFAULT_GROUP_THRESHOLD = _threshold_cfg.get("default_group", 5)
_DEFAULT_PRIVATE_THRESHOLD = _threshold_cfg.get("default_private", 3)
# config 中按对象设置的阈值
_CONFIG_THRESHOLDS = {str(k): v for k, v in _threshold_cfg.items() if k not in ("default_group", "default_private")}


class ChatSession:
    """单个聊天对象（群或私聊）的全部运行时状态"""

    def __init__(self, chat_id: int, is_group: bool):
        self.chat_id = chat_id
        self.is_group = is_group
        self.memory = PersonaMemory(chat_id)

        # —— 每个聊天独立的开关 ——
        self.enabled: bool = True
        self.use_custom_prompt: bool = False
        self.bypass_master_detection: bool = False

        # —— 回复阈值（None 表示使用默认值） ——
        self._reply_threshold: Optional[int] = None

        # —— 处理状态 ——
        self._processing: bool = False
        self._should_cancel: bool = False
        self._lock = threading.Lock()

        # —— bypass 自动恢复定时器 ——
        self._bypass_timer: Optional[threading.Timer] = None

    # ──────── 阈值 ────────

    @property
    def reply_threshold(self) -> int:
        """获取阈值，优先级：运行时 > config按对象 > config默认值"""
        if self._reply_threshold is not None:
            return self._reply_threshold
        key = f"{'group' if self.is_group else 'private'}_{self.chat_id}"
        config_val = _CONFIG_THRESHOLDS.get(key)
        if config_val is not None:
            return config_val
        return _DEFAULT_GROUP_THRESHOLD if self.is_group else _DEFAULT_PRIVATE_THRESHOLD

    @reply_threshold.setter
    def reply_threshold(self, value: Optional[int]):
        self._reply_threshold = value

    def reset_reply_threshold(self):
        """重置为默认阈值"""
        self._reply_threshold = None

    # ──────── 处理状态 ────────

    @property
    def is_processing(self) -> bool:
        with self._lock:
            return self._processing

    @is_processing.setter
    def is_processing(self, value: bool):
        with self._lock:
            self._processing = value

    @property
    def should_cancel(self) -> bool:
        with self._lock:
            return self._should_cancel

    @should_cancel.setter
    def should_cancel(self, value: bool):
        with self._lock:
            self._should_cancel = value

    # ──────── bypass master detection ────────

    def set_bypass_master_detection(self, enabled: bool, minutes: int = 30):
        """设置 bypass，可选自动恢复"""
        if self._bypass_timer:
            self._bypass_timer.cancel()
            self._bypass_timer = None

        self.bypass_master_detection = enabled

        if enabled and minutes > 0:
            def _restore():
                self.bypass_master_detection = False
                print(f"[MasterDetection] chat {self.chat_id} 主人检测已自动恢复（{minutes}分钟到期）")

            self._bypass_timer = threading.Timer(minutes * 60, _restore)
            self._bypass_timer.daemon = True
            self._bypass_timer.start()
            print(f"[MasterDetection] chat {self.chat_id} 主人检测已关闭，将在 {minutes} 分钟后自动恢复")
        else:
            print(f"[MasterDetection] chat {self.chat_id} 主人检测已{'关闭' if enabled else '恢复'}")


class ChatManager:
    """管理所有 ChatSession 以及全局去重状态"""

    def __init__(self):
        self._sessions: dict[int, ChatSession] = {}
        self._lock = threading.Lock()
        self._recent_msg_ids: deque = deque(maxlen=_MAX_RECENT_MSG_IDS)
        self._msg_ids_lock = threading.Lock()

        # 启动时从文件加载运行时阈值
        self._saved_thresholds = self._load_thresholds()

    # ──────── Session 管理 ────────

    def get_or_create(self, chat_id: int, is_group: bool) -> ChatSession:
        """获取或创建 ChatSession，首次创建时加载持久化阈值"""
        with self._lock:
            session = self._sessions.get(chat_id)
            if session is None:
                session = ChatSession(chat_id, is_group)
                # 恢复持久化的阈值
                key = f"{'group' if is_group else 'private'}_{chat_id}"
                saved = self._saved_thresholds.get(key)
                if saved is not None:
                    session._reply_threshold = saved
                self._sessions[chat_id] = session
            return session

    def get(self, chat_id: int) -> Optional[ChatSession]:
        with self._lock:
            return self._sessions.get(chat_id)

    # ──────── 消息去重 ────────

    def is_duplicate(self, msg_id) -> bool:
        with self._msg_ids_lock:
            if msg_id in self._recent_msg_ids:
                return True
            self._recent_msg_ids.append(msg_id)
            return False

    # ──────── 阈值持久化 ────────

    def save_threshold(self, session: ChatSession):
        """将某个 session 的阈值写入文件"""
        key = f"{'group' if session.is_group else 'private'}_{session.chat_id}"
        if session._reply_threshold is not None:
            self._saved_thresholds[key] = session._reply_threshold
        elif key in self._saved_thresholds:
            del self._saved_thresholds[key]
        self._persist_thresholds()

    def _load_thresholds(self) -> dict:
        try:
            if os.path.exists(_THRESHOLDS_FILE):
                with open(_THRESHOLDS_FILE, "r", encoding="utf-8") as f:
                    return {str(k): v for k, v in json.load(f).items()}
        except Exception:
            pass
        return {}

    def _persist_thresholds(self):
        with open(_THRESHOLDS_FILE, "w", encoding="utf-8") as f:
            json.dump(self._saved_thresholds, f, ensure_ascii=False, indent=2)

    # ──────── 清理 ────────

    def cleanup_old_sessions(self, max_age_hours: float = 24):
        """清理长时间没有活动的 session"""
        import time
        current = time.time()
        to_delete = []
        with self._lock:
            for chat_id, session in self._sessions.items():
                if session.memory.last_spoken_time:
                    diff_hours = (current - session.memory.last_spoken_time.timestamp()) / 3600
                    if diff_hours > max_age_hours:
                        to_delete.append(chat_id)
            for chat_id in to_delete:
                del self._sessions[chat_id]
                print(f"Cleaned up session for chat {chat_id}")


# 全局单例
chat_manager = ChatManager()