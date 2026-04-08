from flask import Flask, jsonify, request
from qq_msg import parse_msg
from database import DatabaseManager
from config import CONFIG
import qq_msg
import threading
from functools import wraps
import requests as http_requests

app = Flask(__name__)

_msg_cfg = CONFIG["message_server"]
_master_user_id = CONFIG["master_user_id"]
debug = _msg_cfg.get("debug", False)
db = DatabaseManager()
target_groups = _msg_cfg["target_groups"]
target_users = _msg_cfg["target_users"]

# Database API token (从 config 读取)
_db_api_token = _msg_cfg.get("db_api_token", "")


# ──────────────── Token 验证装饰器 ────────────────

def require_db_token(f):
    """验证数据库 API 请求的 token"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not _db_api_token:
            # 如果没有配置 token，允许访问（向后兼容）
            return f(*args, **kwargs)

        auth_header = request.headers.get('Authorization', '')

        # 支持 "Bearer <token>" 格式
        if auth_header.startswith('Bearer '):
            provided_token = auth_header[7:]
        else:
            provided_token = auth_header

        if provided_token != _db_api_token:
            return jsonify({
                "status": "error",
                "error": "Unauthorized: Invalid or missing token"
            }), 401

        return f(*args, **kwargs)
    return decorated_function

# === 运行状态 ===
_running_lock = threading.Lock()
_chat_bot_run = True

def _set_running(running):
    global _chat_bot_run
    with _running_lock:
        _chat_bot_run = running

def notify_server_to_process(event_data):
    """
    通知 server.py 处理新消息（非阻塞）
    """
    def _notify():
        msg_id = event_data.get('message_id')
        try:
            print(f"[NOTIFY] 通知 server 处理消息: msg_id={msg_id}")
            resp = http_requests.post(
                "http://127.0.0.1:5001/internal/process",
                json=event_data,
                timeout=10  # 增加超时时间，适应 server 的延迟
            )
            print(f"[NOTIFY] server 响应成功: msg_id={msg_id}, status={resp.status_code}")
        except http_requests.exceptions.Timeout:
            # 超时不是错误，server 可能还在处理
            print(f"[NOTIFY] 通知 server 超时（正常，server 仍在处理）: msg_id={msg_id}")
        except http_requests.exceptions.ConnectionError as e:
            # 连接失败，server 可能没启动
            print(f"[NOTIFY] 无法连接到 server (port 5001): {e}")
        except Exception as e:
            # 其他错误
            print(f"[NOTIFY] 通知 server 失败: {e}")

    # 非阻塞通知
    threading.Thread(target=_notify, daemon=True).start()



@app.route('/', methods=['POST'])
def handle_nc_event():
    event_data = request.json
    post_type = event_data.get('post_type')
    # print(event_data)
    # master 自己手动发送的消息（message_sent 事件），只存入数据库，不触发 AI 处理
    if post_type == 'message_sent':
        # 检查是否为 trigger cmd（开关机器人）
        if event_data.get('raw_message') == CONFIG["chatbot_server"].get("trun_on_cmd") or \
           event_data.get('raw_message') == CONFIG["chatbot_server"].get("trun_off_cmd"):
            notify_server_to_process(event_data)
            print("is trigger command")
            return jsonify({'status': 'ok'})
        # 检查是否为 / 开头的命令（遍历所有 segments，不依赖固定索引）
        try:
            for seg in event_data.get("message", []):
                if isinstance(seg, dict) and seg.get('type') == 'text':
                    if seg.get('data', {}).get('text', '').replace(" ", "").startswith("/"):
                        notify_server_to_process(event_data)
                        print("is command")
                        return jsonify({'status': 'ok'})
                    break
        except Exception:
            print("not command")
        is_sent_to_group = (event_data.get('message_type') == 'group' and
                            event_data.get('group_id') in target_groups)
        # 私聊 message_sent 中接收者在 target_id，不是 user_id
        is_sent_to_user = (event_data.get('message_type') == 'private' and
                           event_data.get('target_id') in target_users)
        if is_sent_to_group or is_sent_to_user:
            msg = parse_msg(event_data, quick_mode=False)
            if msg:
                # 私聊时 target_id 是接收者，修正 peer_id
                if not msg.is_group:
                    msg.peer_id = event_data.get('target_id')
                db.add_msg(msg)
        return jsonify({'status': 'ok'})

    # 判断是否是目标消息
    is_target_private = (event_data.get('message_type') == 'private' and
                         event_data.get('user_id') in target_users)
    is_master_private = (event_data.get('message_type') == 'private' and
                         event_data.get('sender', {}).get('user_id') == _master_user_id)
    is_target_group = (event_data.get('message_type') == 'group' and
                       event_data.get('group_id') in target_groups)

    is_target = is_target_private or is_master_private or is_target_group

    if not is_target:
        # 非目标消息，不处理
        return jsonify({'status': 'ok'})

    # 检查是否正在运行
    with _running_lock:
        if not _chat_bot_run:
            return jsonify({'status': 'ok', 'running': False})

    # 同步解析消息（包括图片）
    msg = parse_msg(event_data, quick_mode=False)
    if msg:
        if debug:
            print(msg.to_str()[:100])
        db.add_msg(msg)  # 此时消息内容已包含完整图片描述

        # 解析完成后通知 server.py 处理
        notify_server_to_process(event_data)

    return jsonify({'status': 'ok', 'running': True})

# ──────────────── Database API Routes ────────────────

@app.route('/db/add_msg', methods=['POST'])
@require_db_token
def db_add_msg():
    """API endpoint: Add a message to database."""
    try:
        from database import MessageModel
        data = request.json
        msg = MessageModel(
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
        db.add_msg(msg)
        return jsonify({"status": "ok"})
    except Exception as e:
        return jsonify({"status": "error", "error": str(e)}), 500


@app.route('/db/get_latest_by_count', methods=['POST'])
@require_db_token
def db_get_latest_by_count():
    """API endpoint: Get latest messages by count."""
    try:
        data = request.json
        target_id = data["target_id"]
        is_group = data["is_group"]
        max_msg = data.get("max_msg", 20)

        messages = db.get_latest_messages_by_count(target_id, is_group, max_msg)
        return jsonify({
            "status": "ok",
            "messages": [
                {
                    "msg_id": m.msg_id,
                    "user_id": m.user_id,
                    "group_id": m.group_id,
                    "sender_name": m.sender_name,
                    "sender_card": m.sender_card,
                    "context": m.context,
                    "timestamp": m.timestamp,
                    "is_ai": m.is_ai,
                    "is_group": m.is_group,
                    "peer_id": m.peer_id
                }
                for m in messages
            ]
        })
    except Exception as e:
        return jsonify({"status": "error", "error": str(e)}), 500


@app.route('/db/get_latest_by_time', methods=['POST'])
@require_db_token
def db_get_latest_by_time():
    """API endpoint: Get latest messages by time."""
    try:
        data = request.json
        target_id = data["target_id"]
        is_group = data.get("is_group", True)
        minutes = data.get("minutes", 60)

        messages = db.get_latest_messages_by_time(target_id, is_group, minutes)
        return jsonify({
            "status": "ok",
            "messages": [
                {
                    "msg_id": m.msg_id,
                    "user_id": m.user_id,
                    "group_id": m.group_id,
                    "sender_name": m.sender_name,
                    "sender_card": m.sender_card,
                    "context": m.context,
                    "timestamp": m.timestamp,
                    "is_ai": m.is_ai,
                    "is_group": m.is_group,
                    "peer_id": m.peer_id
                }
                for m in messages
            ]
        })
    except Exception as e:
        return jsonify({"status": "error", "error": str(e)}), 500


@app.route('/db/update_is_ai', methods=['POST'])
@require_db_token
def db_update_is_ai():
    """API endpoint: Update is_ai flag."""
    try:
        data = request.json
        msg_id = data["msg_id"]
        is_ai = data["is_ai"]

        row_count = db.update_is_ai(msg_id, is_ai)
        return jsonify({"status": "ok", "row_count": row_count})
    except Exception as e:
        return jsonify({"status": "error", "error": str(e)}), 500


@app.route('/db/get_latest_by_sender', methods=['POST'])
@require_db_token
def db_get_latest_by_sender():
    """API endpoint: Get latest messages by sender_id."""
    try:
        data = request.json
        sender_id = data["sender_id"]
        limit = data.get("limit", 20)

        messages = db.get_latest_messages_by_sender_id(sender_id, limit)
        return jsonify({
            "status": "ok",
            "messages": [
                {
                    "msg_id": m.msg_id,
                    "user_id": m.user_id,
                    "group_id": m.group_id,
                    "sender_name": m.sender_name,
                    "sender_card": m.sender_card,
                    "context": m.context,
                    "timestamp": m.timestamp,
                    "is_ai": m.is_ai,
                    "is_group": m.is_group,
                    "peer_id": m.peer_id
                }
                for m in messages
            ]
        })
    except Exception as e:
        return jsonify({"status": "error", "error": str(e)}), 500


@app.route('/db/delete_old', methods=['POST'])
@require_db_token
def db_delete_old():
    """API endpoint: Delete old messages."""
    try:
        data = request.json
        older_than_minutes = data.get("older_than_minutes", 10080)

        deleted_count = db.delete_old_messages(older_than_minutes)
        return jsonify({"status": "ok", "deleted_count": deleted_count})
    except Exception as e:
        return jsonify({"status": "error", "error": str(e)}), 500


@app.route('/run', methods=['POST'])
def switch_msg_server():
    status = request.json
    cmd = status.get("status", "")
    global _chat_bot_run

    if cmd == "on":
        with _running_lock:
            if _chat_bot_run:
                print("already running")
                return jsonify({"status":"ok"})

        _set_running(True)
        print("turn on")
        # 导入历史消息
        for target_group in target_groups:
            try:
                real_msgs = qq_msg.get_group_message(target_group, 20).get('data', {}).get('messages', [])
                db_msgs = db.get_latest_messages_by_count(target_group, True, 1)

                if db_msgs:
                    db_msg = db_msgs[-1]
                    import_data_flg = False
                    for real_msg in real_msgs:
                        if import_data_flg:
                            db.add_msg(parse_msg(real_msg))
                        if str(real_msg.get('message_id')) == str(db_msg.msg_id):
                            import_data_flg = True
                    if not import_data_flg:
                        for real_msg in real_msgs:
                            db.add_msg(parse_msg(real_msg))
                else:
                    for real_msg in real_msgs:
                        db.add_msg(parse_msg(real_msg))
            except Exception as e:
                print(f"导入群{target_group}历史消息失败: {e}")

        return jsonify({'status': 'ok'})

    elif cmd == "off":
        _set_running(False)
        print("turn off")
        return jsonify({'status': 'ok'})

    else:
        return jsonify({'status': 'error', 'message': '未知命令'}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5002, debug=debug)
