import sqlite3
import json
import time
from datetime import datetime
from typing import List
from config import CONFIG

_master_user_id = CONFIG["master_user_id"]

class MessageModel:
    """
    Model representing a chat message.
    """
    def __init__(self, msg_id, sender_id, group_id, sender_name, sender_card, content, timestamp, is_ai, is_group, peer_id=None):
        self.msg_id = msg_id #消息id
        self.user_id = sender_id #发送者id
        self.group_id = group_id #群聊id
        self.sender_name = sender_name #发送者昵称
        self.sender_card = sender_card #发送者名片
        self.context = content #消息内容
        self.timestamp = str(timestamp) #发送时间
        self.is_ai = is_ai #是否为ai自己发送的消息
        self.is_group = is_group #是否为群聊消息
        self.peer_id = peer_id  # 私聊对话对象的 user_id（始终为对方的 ID）
        self.is_master = False #是否是被数字化身的本人说话
        if sender_id == _master_user_id and not self.is_ai:
            self.is_master = True

    def to_str(self, index=None):
        sender = ""
        if self.user_id == _master_user_id:
            sender = f"{CONFIG['master_name']}(我)"
        elif self.is_group and self.sender_card:
            sender = f"{self.sender_card}({self.sender_name})"
        else:
            sender = self.sender_name
        prefix = f"[#{index}]" if index is not None else ""
        return f"{prefix}[{self.timestamp[8:10]}:{self.timestamp[10:12]}:{self.timestamp[12:14]}]{sender}:{self.context}"

    def to_tuple(self):
        """Returns a tuple for SQLite insertion."""
        # Ensure timestamp is stored as integer for compatibility/sorting if it's a digit string
        ts = self.timestamp
        if isinstance(ts, str) and ts.isdigit():
            ts = int(ts)
            
        return (
            self.msg_id,
            ts,
            self.group_id,
            self.user_id,
            self.sender_name,
            self.sender_card,
            self.context,
            1 if self.is_ai else 0,
            1 if self.is_group else 0,
            self.peer_id
        )


class DatabaseManager:
    """
    Handles SQLite database operations for chat history.
    """
    def __init__(self, db_path="chat_history.db"):
        self.db_path = db_path
        self._init_table()

    def _get_conn(self):
        return sqlite3.connect(self.db_path)

    def _init_table(self):
        """Initializes the table if it doesn't exist."""
        # msg_id is now the Primary Key
        # timestamp is no longer PK
        # id is removed, replaced by group_id and user_id
        create_sql = """
        CREATE TABLE IF NOT EXISTS messages (
            msg_id INTEGER PRIMARY KEY,
            timestamp INTEGER,
            group_id INTEGER,
            user_id INTEGER,
            sender_name TEXT,
            sender_card TEXT,
            context TEXT,
            is_ai INTEGER,
            is_group INTEGER,
            peer_id INTEGER
        );
        """
        # Index for efficient querying
        create_index_group = "CREATE INDEX IF NOT EXISTS idx_group ON messages (group_id, is_group);"
        create_index_user = "CREATE INDEX IF NOT EXISTS idx_user ON messages (user_id, is_group);"
        create_index_peer = "CREATE INDEX IF NOT EXISTS idx_peer ON messages (peer_id, is_group);"

        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute(create_sql)
        cursor.execute(create_index_group)
        cursor.execute(create_index_user)
        # 迁移：如果旧表没有 peer_id 列，自动添加
        try:
            cursor.execute("ALTER TABLE messages ADD COLUMN peer_id INTEGER")
        except Exception:
            pass  # 列已存在，忽略
        # 每次启动都回填 peer_id 为 NULL 的私聊记录
        # 非 AI 消息：peer_id = user_id（发送者就是对方）
        cursor.execute("UPDATE messages SET peer_id = user_id WHERE is_group = 0 AND is_ai = 0 AND peer_id IS NULL")
        # AI 消息：从同一对话中最近的非 AI 消息推断 peer_id
        cursor.execute("""
            UPDATE messages SET peer_id = (
                SELECT m2.user_id FROM messages m2
                WHERE m2.is_group = 0 AND m2.is_ai = 0 AND m2.user_id != messages.user_id
                AND m2.timestamp <= messages.timestamp
                ORDER BY m2.timestamp DESC LIMIT 1
            )
            WHERE is_group = 0 AND is_ai = 1 AND peer_id IS NULL
        """)
        # peer_id 列确保存在后再创建索引
        cursor.execute(create_index_peer)
        conn.commit()
        conn.close()

    def add_msg(self, msg: MessageModel):
        """Inserts a message into the database."""
        sql = """
        INSERT INTO messages (msg_id, timestamp, group_id, user_id, sender_name, sender_card, context, is_ai, is_group, peer_id)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        try:
            conn = self._get_conn()
            cursor = conn.cursor()
            cursor.execute(sql, msg.to_tuple())
            conn.commit()
        except sqlite3.IntegrityError as e:
            print(f"Error inserting message {msg.msg_id}: {e}")
        finally:
            if 'conn' in locals():
                conn.close() # type: ignore

    def _row_to_model(self, row):
        """Helper to convert a DB row to a MessageModel."""
        return MessageModel(
            msg_id=row[0],
            sender_id=row[3],
            group_id=row[2],
            sender_name=row[4],
            sender_card=row[5],
            content=row[6],
            timestamp=row[1],
            is_ai=bool(row[7]),
            is_group=bool(row[8]),
            peer_id=row[9] if len(row) > 9 else None
        )

    def get_latest_messages_by_count(self, target_id, is_group, max_msg=20) -> List[MessageModel]:
        """
        Get latest X messages for specific ID and type.
        target_id: group_id if is_group, else user_id
        """
        if is_group:
            sql = """
            SELECT msg_id, timestamp, group_id, user_id, sender_name, sender_card, context, is_ai, is_group, peer_id
            FROM messages
            WHERE group_id = ? AND is_group = 1
            ORDER BY timestamp DESC
            LIMIT ?
            """
        else:
            sql = """
            SELECT msg_id, timestamp, group_id, user_id, sender_name, sender_card, context, is_ai, is_group, peer_id
            FROM messages
            WHERE (peer_id = ? OR (peer_id IS NULL AND user_id = ?)) AND is_group = 0
            ORDER BY timestamp DESC
            LIMIT ?
            """

        conn = self._get_conn()
        cursor = conn.cursor()
        if is_group:
            cursor.execute(sql, (target_id, max_msg))
        else:
            cursor.execute(sql, (target_id, target_id, max_msg))
        rows = cursor.fetchall()
        conn.close()

        results = [self._row_to_model(r) for r in rows]
        # Reverse to get chronological order (Old -> New)
        return results[::-1]

    def get_latest_messages_by_time(self, target_id, is_group = True, minutes=60):
        """
        Get all messages from (Now - Minutes) to Now.
        """
        # Calculate cutoff timestamp
        # Assuming timestamp is stored as yyyyMMddHHmmss (integer)
        cutoff_dt = datetime.fromtimestamp(time.time() - (minutes * 60))
        cutoff_str = cutoff_dt.strftime("%Y%m%d%H%M%S")
        cutoff_ts = int(cutoff_str)

        if is_group:
             sql = """
            SELECT msg_id, timestamp, group_id, user_id, sender_name, sender_card, context, is_ai, is_group, peer_id
            FROM messages
            WHERE group_id = ? AND is_group = 1 AND timestamp >= ?
            ORDER BY timestamp ASC
            """
        else:
             sql = """
            SELECT msg_id, timestamp, group_id, user_id, sender_name, sender_card, context, is_ai, is_group, peer_id
            FROM messages
            WHERE (peer_id = ? OR (peer_id IS NULL AND user_id = ?)) AND is_group = 0 AND timestamp >= ?
            ORDER BY timestamp ASC
            """

        conn = self._get_conn()
        cursor = conn.cursor()
        if is_group:
            cursor.execute(sql, (target_id, cutoff_ts))
        else:
            cursor.execute(sql, (target_id, target_id, cutoff_ts))
        rows = cursor.fetchall()
        conn.close()

        return [self._row_to_model(r) for r in rows]

    def update_is_ai(self, msg_id, is_ai):
        """
        Updates the is_ai status of a specific message by msg_id.
        Returns number of modified rows.
        """
        sql = "UPDATE messages SET is_ai = ? WHERE msg_id = ?"
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute(sql, (1 if is_ai else 0, msg_id))
        row_count = cursor.rowcount
        conn.commit()
        conn.close()
        return row_count

    def update_message_content(self, msg_id, new_content):
        """
        Updates the content of a specific message by msg_id.
        Returns number of modified rows.
        """
        sql = "UPDATE messages SET context = ? WHERE msg_id = ?"
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute(sql, (new_content, msg_id))
        row_count = cursor.rowcount
        conn.commit()
        conn.close()
        return row_count

    def get_latest_messages_by_sender_id(self, sender_id, limit=20) -> List[MessageModel]:
        """
        Get latest X messages for a specific sender_id.
        """
        sql = """
        SELECT msg_id, timestamp, group_id, user_id, sender_name, sender_card, context, is_ai, is_group, peer_id
        FROM messages
        WHERE user_id = ?
        ORDER BY timestamp DESC
        LIMIT ?
        """
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute(sql, (sender_id, limit))
        rows = cursor.fetchall()
        conn.close()

        results = [self._row_to_model(r) for r in rows]
        return results[::-1]

    def delete_old_messages(self, older_than_minutes=10080):
        """
        Delete all messages older than X minutes.
        """
        cutoff_dt = datetime.fromtimestamp(time.time() - (older_than_minutes * 60))
        cutoff_str = cutoff_dt.strftime("%Y%m%d%H%M%S")
        cutoff_ts = int(cutoff_str)

        sql = "DELETE FROM messages WHERE timestamp < ?"
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute(sql, (cutoff_ts,))
        deleted_count = cursor.rowcount
        conn.commit()
        conn.close()
        return deleted_count

class NetDatabaseManager:
    """
    Network-based database manager that uses HTTP requests to access database operations.
    Maintains the same interface as DatabaseManager for drop-in compatibility.
    """
    def __init__(self, base_url=None):
        from config import CONFIG
        if base_url is None:
            # Default to msg_server (port 5002) instead of main server
            base_url = "http://127.0.0.1:5002"
        self.base_url = base_url.rstrip("/")
        # 使用专门的数据库 API token（与 msg_server 保持一致）
        self.token = CONFIG.get("message_server", {}).get("db_api_token", "")

    def _request(self, endpoint: str, method: str = "POST", data: dict = None):
        """Make HTTP request to the database API."""
        import requests
        url = f"{self.base_url}/{endpoint}"
        headers = {
            "Content-Type": "application/json",
        }
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"

        try:
            if method == "POST":
                response = requests.post(url, json=data, headers=headers, timeout=10)
            else:
                response = requests.get(url, params=data, headers=headers, timeout=10)

            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"[NetDatabaseManager] Request failed: {e}")
            return None

    def _dict_to_model(self, data: dict) -> MessageModel:
        """Convert dict from API to MessageModel."""
        return MessageModel(
            msg_id=data["msg_id"],
            sender_id=data["user_id"],
            group_id=data["group_id"],
            sender_name=data["sender_name"],
            sender_card=data["sender_card"],
            content=data["context"],
            timestamp=data["timestamp"],
            is_ai=data["is_ai"],
            is_group=data["is_group"],
            peer_id=data.get("peer_id")
        )

    def _model_to_dict(self, msg: MessageModel) -> dict:
        """Convert MessageModel to dict for API."""
        return {
            "msg_id": msg.msg_id,
            "user_id": msg.user_id,
            "group_id": msg.group_id,
            "sender_name": msg.sender_name,
            "sender_card": msg.sender_card,
            "context": msg.context,
            "timestamp": msg.timestamp,
            "is_ai": msg.is_ai,
            "is_group": msg.is_group,
            "peer_id": msg.peer_id
        }

    def add_msg(self, msg: MessageModel):
        """Inserts a message into the database via API."""
        data = self._model_to_dict(msg)
        result = self._request("db/add_msg", data=data)
        if result and result.get("status") != "ok":
            print(f"[NetDatabaseManager] add_msg failed: {result.get('error')}")

    def get_latest_messages_by_count(self, target_id, is_group, max_msg=20) -> List[MessageModel]:
        """Get latest X messages for specific ID and type via API."""
        data = {
            "target_id": target_id,
            "is_group": is_group,
            "max_msg": max_msg
        }
        result = self._request("db/get_latest_by_count", data=data)
        if result and result.get("status") == "ok":
            return [self._dict_to_model(m) for m in result.get("messages", [])]
        return []

    def get_latest_messages_by_time(self, target_id, is_group=True, minutes=60) -> List[MessageModel]:
        """Get all messages from (Now - Minutes) to Now via API."""
        data = {
            "target_id": target_id,
            "is_group": is_group,
            "minutes": minutes
        }
        result = self._request("db/get_latest_by_time", data=data)
        if result and result.get("status") == "ok":
            return [self._dict_to_model(m) for m in result.get("messages", [])]
        return []

    def update_is_ai(self, msg_id, is_ai):
        """Updates the is_ai status of a specific message by msg_id via API."""
        data = {
            "msg_id": msg_id,
            "is_ai": is_ai
        }
        result = self._request("db/update_is_ai", data=data)
        if result and result.get("status") == "ok":
            return result.get("row_count", 0)
        return 0

    def get_latest_messages_by_sender_id(self, sender_id, limit=20) -> List[MessageModel]:
        """Get latest X messages for a specific sender_id via API."""
        data = {
            "sender_id": sender_id,
            "limit": limit
        }
        result = self._request("db/get_latest_by_sender", data=data)
        if result and result.get("status") == "ok":
            return [self._dict_to_model(m) for m in result.get("messages", [])]
        return []

    def delete_old_messages(self, older_than_minutes=10080):
        """Delete all messages older than X minutes via API."""
        data = {
            "older_than_minutes": older_than_minutes
        }
        result = self._request("db/delete_old", data=data)
        if result and result.get("status") == "ok":
            return result.get("deleted_count", 0)
        return 0


# ──────────────── 全局数据库实例（根据 config 自动选择） ────────────────

def get_db_instance():
    """
    根据配置返回数据库管理器实例（单例模式）。
    在 CONFIG 中设置 "use_net_database": true 使用网络版本。
    """
    use_net = CONFIG["chatbot_server"].get("use_net_database",False)

    if use_net:
        return NetDatabaseManager()
    else:
        return DatabaseManager()


# 模块级单例实例，导入时自动创建
db = get_db_instance()


if __name__ == "__main__":
    import os
    
    # Setup test DB
    test_db_path = "chat_history.db"
    if os.path.exists(test_db_path):
        os.remove(test_db_path)
        
    db = DatabaseManager(test_db_path)
    print("=== Database Initialized ===")

    current_time_str = datetime.now().strftime("%Y%m%d%H%M%S")
    
    # 1. Test Add & Get by Count
    print("\n--- Test: Add Message & Get by Count ---")
    db.add_msg(MessageModel(
        msg_id=101,
        sender_id=222,
        group_id=1001,
        sender_name="User1",
        sender_card="Card1",
        content="Hello World",
        timestamp=current_time_str,
        is_ai=False,
        is_group=True
    ))

    # Add second message slightly later (simulated by just adding to timestamp integer)
    next_time = int(current_time_str) + 1
    db.add_msg(MessageModel(
        msg_id=102,
        sender_id=333,
        group_id=1001,
        sender_name="User2",
        sender_card="Card2",
        content="Second Message",
        timestamp=str(next_time),
        is_ai=False,
        is_group=True
    ))

    msgs_count = db.get_latest_messages_by_count(1001, True, 10)
    print(f"Retrieved {len(msgs_count)} messages (Expected 2)")
    for m in msgs_count:
        print(m)

    # 2. Test Get by Time
    print("\n--- Test: Get by Time ---")
    # Add a message from the past (fake timestamp)
    past_time = int(current_time_str) - 5000 # Just subtract to make it smaller
    db.add_msg(MessageModel(
        msg_id=99,
        sender_id=444,
        group_id=1001,
        sender_name="OldUser",
        sender_card="",
        content="Old Message",
        timestamp=str(past_time),
        is_ai=False,
        is_group=True
    ))
    
    # Check retrieval
    all_msgs = db.get_latest_messages_by_count(1001, True, 10)
    print(f"All messages count: {len(all_msgs)} (Expected 3)")

    # # Clean up
    # if os.path.exists(test_db_path):
    #     os.remove(test_db_path)
