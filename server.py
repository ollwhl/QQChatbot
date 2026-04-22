from flask import Flask, jsonify, request
from qq_msg import send_group_message, get_group_message, get_private_message, send_private_message, parse_msg
import AI_agent
import time
import random
import threading
from database import db
from config import CONFIG
import command
from status import chat_manager
from logger import group_chat_logger, private_chat_logger
import requests
import json
import signal
import sys
import atexit
app = Flask(__name__)

_server_cfg = CONFIG["chatbot_server"]
debug = _server_cfg.get("debug", False)

def validate_config():
    """验证配置是否完整"""
    required_keys = ["target_groups", "target_users", "NAPCAT_HOST", "NAPCAT_TOKEN"]
    for key in required_keys:
        if key not in _server_cfg:
            raise ValueError(f"Missing required config key: {key}")
    
    if not isinstance(_server_cfg["target_groups"], list):
        raise ValueError("target_groups must be a list")
    
    if not isinstance(_server_cfg["target_users"], list):
        raise ValueError("target_users must be a list")
    
    print(f"Config validated: {len(_server_cfg['target_groups'])} target groups, {len(_server_cfg['target_users'])} target users")

def turn_msg_server_on(max_retries=3, retry_delay=2):
    url = "http://127.0.0.1:5002/run"
    payload = json.dumps({"status": "on"})
    headers = {'Content-Type': 'application/json'}

    for attempt in range(max_retries):
        try:
            resp = requests.post(url, headers=headers, data=payload, timeout=5)
            if resp.json().get("status") == "ok":
                print("msg server is now running")
                return True
            else:
                print("something going wrong with msg server")
                return False
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"msg server not found (attempt {attempt + 1}/{max_retries}), retrying in {retry_delay}s...")
                time.sleep(retry_delay)
            else:
                print("msg server not found (all retries exhausted)")
                return False
    return False

def turn_msg_server_off():
    url = "http://127.0.0.1:5002/run"
    payload = json.dumps({"status": "off"})
    headers = {'Content-Type': 'application/json'}
    try:
        resp = requests.post(url, headers=headers, data=payload, timeout=5)
        if resp.json().get("status") == "ok":
            print("msg server is now shut down")
            return True
        else:
            print("something going wrong with msg server")
            return False
    except Exception:
        print("msg server not found")
        return False
    
def signal_handler(signum, frame):
    """信号处理函数"""
    turn_msg_server_off()
    # 正常退出程序
    sys.exit(0)
# 注册信号处理器
signal.signal(signal.SIGINT, signal_handler)   # Ctrl+C
signal.signal(signal.SIGTERM, signal_handler)  # kill 命令

# 使用 atexit 注册（在正常退出时也会执行）
@atexit.register
def at_exit():
    turn_msg_server_off()
    
    

def send_messages_with_delay(send_func, target_id, reply, logger):
    """
    发送消息（支持单条和多条）

    Args:
        send_func: 发送函数（send_group_message 或 send_private_message）
        target_id: 目标 ID（群号或用户 ID）
        reply: 回复内容（str 或 List[str]）
        logger: 日志记录器
    """
    if not reply:
        return

    # 统一转为列表，并按中文逗号拆分
    if isinstance(reply, str):
        parts = [p.strip() for p in reply.split("，") if p.strip()]
    elif isinstance(reply, list):
        parts = []
        for item in reply:
            parts.extend(p.strip() for p in item.split("，") if p.strip())
    else:
        return

    logger.info(f"Sending {len(parts)} messages...")
    for i, msg in enumerate(parts):
        send_func(target_id, msg)
        logger.info(f"Sent message {i+1}/{len(parts)}: {msg}")

        if i < len(parts) - 1:
            delay = random.uniform(1.0, 2.0)
            time.sleep(delay)

def _process_event(event_data):
    """
    核心事件处理逻辑（被 / 和 /internal/process 共用）
    """
    logger = group_chat_logger if event_data.get('message_type') == 'group' else private_chat_logger
    target_groups = _server_cfg["target_groups"]
    target_users = _server_cfg["target_users"]

    # 去重检查：防止重复处理同一条消息
    msg_id = event_data.get('message_id')
    print(f"[EVENT] type={event_data.get('message_type')}, id={msg_id}, from={event_data.get('user_id') or event_data.get('group_id')}")
    if msg_id and chat_manager.is_duplicate(msg_id):
        print(f"[SKIP] 重复请求已忽略: msg_id={msg_id}")
        logger.info(f"重复请求已忽略: msg_id={msg_id}")
        return jsonify({'status': 'ok', 'reason': 'duplicate'})
    # 调试日志
    if debug:
        print(f"[DEBUG] Received event: msg_type={event_data.get('message_type')}, msg_id={msg_id}, from={event_data.get('user_id') or event_data.get('group_id')}")

    # 获取 session
    is_group = event_data.get('message_type') == 'group'
    chat_target = event_data.get('group_id') if is_group else event_data.get('user_id')
    session = chat_manager.get_or_create(chat_target, is_group) if chat_target else None

    cmd_result = command.handle_cmd(event_data, session)
    if cmd_result is not None:
        return jsonify({
            'status': 'ok',
        })
    # trigger cmd: 按聊天对象调整回复阈值
    _THRESHOLD_STEP = 2  # 每次调整的步长
    raw_msg = event_data.get('raw_message')
    if raw_msg in (CONFIG["chatbot_server"].get("trun_on_cmd"), CONFIG["chatbot_server"].get("trun_off_cmd")):
        if session:
            current = session.reply_threshold
            if raw_msg == CONFIG["chatbot_server"].get("trun_on_cmd"):
                new_val = max(0, current - _THRESHOLD_STEP)
            else:
                new_val = min(10, current + _THRESHOLD_STEP)
            session.reply_threshold = new_val
            chat_manager.save_threshold(session)
            target_label = f"群{chat_target}" if is_group else f"用户{chat_target}"
            logger.info(f"Reply threshold for {target_label}: {current} -> {new_val}")

    # ——————————————————PRIVATE MESSAGE————————————————————
    if event_data.get('message_type') == 'private':
        if event_data.get('user_id') in target_users:
            target_user = event_data.get('user_id')
            session = chat_manager.get_or_create(target_user, False)
            if debug:
                print(f"[DEBUG] Private message from {target_user}, enabled={session.enabled}")
            if session.enabled:
                # 私聊场景：如果正在处理中，设置取消标志并等待
                if session.is_processing:
                    print(f"私聊 {target_user} 正在处理中，设置取消标志并等待新消息")
                    session.should_cancel = True

                    # 等待一小段时间，让旧请求完成或被取消
                    max_wait = 2.0  # 最多等待2秒
                    wait_interval = 0.1
                    waited = 0
                    while session.is_processing and waited < max_wait:
                        time.sleep(wait_interval)
                        waited += wait_interval

                    # 如果仍在处理，强制清除旧请求状态（可能卡住了）
                    if session.is_processing:
                        logger.warning(f"私聊 {target_user} 等待超时，强制清除旧请求状态，处理新请求")
                        session.is_processing = False
                        # 不拒绝，继续处理新请求

                # 标记为处理中，清除取消标志
                session.is_processing = True
                session.should_cancel = False

                try:
                    time.sleep(random.randint(2, 4))

                    # 处理前再次检查是否应该取消
                    if session.should_cancel:
                        logger.info(f"私聊 {target_user} 请求被取消（有新消息）")
                        return jsonify({'status': 'cancelled'})

                    msg_models = db.get_latest_messages_by_time(target_user, False, 30)
                    has_master_message = False
                    if not session.bypass_master_detection:
                        for msg in msg_models:
                            if msg.is_master:
                                print("Master message detected, AI will not reply.")
                                print(f"消息内容: {msg.to_str()}")
                                has_master_message = True
                                break
                    if len(msg_models) < 20 :
                        msg_models = db.get_latest_messages_by_count(target_user,False,20)
                    if not has_master_message:
                        reply = AI_agent.call_AI_agent(logger, msg_models, session, target_id=target_user)
                        if reply and not debug:
                        # 发送前再次检查是否应该取消
                            if session.should_cancel:
                                logger.info(f"私聊 {target_user} 请求被取消（有新消息），不发送回复")
                                return jsonify({'status': 'cancelled'})

                            send_messages_with_delay(send_private_message, target_user, reply, logger)
                except Exception as e:
                    logger.error(f"Error processing private message: {e}")
                    print(f"Error processing private message: {e}")
                finally:
                    # 处理完成,清除状态
                    session.is_processing = False
                    session.should_cancel = False

    # ——————————————————GROUP MESSAGE————————————————————
    if event_data.get('message_type') == 'group' and event_data.get('group_id') in target_groups and event_data.get('user_id') != event_data.get('self_id'):
        target_group = event_data.get('group_id')
        session = chat_manager.get_or_create(target_group, True)
        if not session.enabled:
            return jsonify({'status': 'ok'})
        print("Received event:", event_data)
        # 检查是否正在处理中,如果是则直接拒绝
        if session.is_processing:
            print(f"拒绝请求: 群 {target_group} 正在处理中")
            return jsonify({
                'status': 'rejected',
                'reason': 'busy'
            })

        # 标记为处理中
        session.is_processing = True
        try:
            time.sleep(random.randint(2, 3))
            # 先尝试获取最近30分钟的消息
            msg_models = db.get_latest_messages_by_time(target_group, True, 30)

            # 如果消息太少，再获取最近20条消息
            if len(msg_models) < 5:
                recent_msg_models = db.get_latest_messages_by_count(target_group, True, 20)
                # 合并消息，去重（按msg_id）
                existing_ids = {msg.msg_id for msg in msg_models}
                for msg in recent_msg_models:
                    if msg.msg_id not in existing_ids:
                        msg_models.append(msg)
                # 按时间排序
                msg_models.sort(key=lambda x: int(x.timestamp) if x.timestamp.isdigit() else 0)


            # 检查是否有master消息
            has_master_message = False
            if not session.bypass_master_detection:
                for msg in msg_models:
                    if msg.is_master:
                        print("Master message detected, AI will not reply.")
                        print(f"消息内容: {msg.to_str()}")
                        has_master_message = True
                        break

            if not has_master_message:
                reply = AI_agent.call_AI_agent(logger, msg_models, session, target_id=target_group)
                if reply and not debug:
                    send_messages_with_delay(send_group_message, target_group, reply, logger)
        except Exception as e:
            logger.error(f"Error processing group message: {e}")
            print(f"Error processing group message: {e}")
        finally:
            # 处理完成,清除状态
            session.is_processing = False

    return jsonify({
        'status': 'ok',
    })

# @app.route('/', methods=['POST'])
# def handle_nc_event():
#     """
#     主webhook路由：直接接收NapCat事件
#     注意：建议只配置 msg_server 接收 NapCat webhook，此路由作为备用
#     """
#     event_data = request.json
#     return _process_event(event_data)

@app.route('/internal/process', methods=['POST'])
def internal_process():
    """
    内部API：接收 msg_server 的通知（串行化保证）
    msg_server 已经存储消息到DB，此处只需处理AI响应逻辑
    """
    event_data = request.json
    return _process_event(event_data)

if __name__ == '__main__':
    # 验证配置
    try:
        validate_config()
    except ValueError as e:
        print(f"Configuration error: {e}")
        sys.exit(1)
    
    # 启动消息服务器
    if not turn_msg_server_on():
        print("Warning: Failed to start message server")
    
    # # 定期清理旧记忆（每6小时一次）
    # import schedule

    # def schedule_cleanup():
    #     schedule.every(6).hours.do(cleanup_old_memorys)
    #     while True:
    #         schedule.run_pending()
    #         time.sleep(60)

    # cleanup_thread = threading.Thread(target=schedule_cleanup, daemon=True)
    # cleanup_thread.start()

    app.run(host='0.0.0.0', port=5001, debug=True)
