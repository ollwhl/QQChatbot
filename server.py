from flask import Flask, jsonify, request
from qq_msg import send_group_message, get_group_message, get_private_message, send_private_message, parse_msg
import AI_agent
import time
import random
import threading
from database import db
from config import CONFIG
import command
from memory import PersonaMemory
import group_pipline
from logger import group_chat_logger, private_chat_logger
import requests
import json
import signal
import sys
import atexit
app = Flask(__name__)

_server_cfg = CONFIG["chatbot_server"]
debug = _server_cfg.get("debug", False)
use_custom_prompt = False
memorys = {}

# 记录被关闭的聊天对象（群号或用户ID），默认全部开启
_disabled_chats = set()
_disabled_chats_mutex = threading.Lock()

# 记录每个聊天目标是否正在处理中
_chat_processing = {}
_chat_processing_mutex = threading.Lock()

# 记录哪些请求应该被取消（私聊专用）
_chat_should_cancel = {}
_chat_cancel_mutex = threading.Lock()

# 记录最近处理过的消息ID，用于去重（保留最近100个）
_recent_msg_ids = []
_recent_msg_ids_mutex = threading.Lock()
_MAX_RECENT_MSG_IDS = 100

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
    
    

def is_chat_processing(chat_id):
    """检查指定聊天是否正在处理中"""
    with _chat_processing_mutex:
        return _chat_processing.get(chat_id, False)

def set_chat_processing(chat_id, processing):
    """设置聊天处理状态"""
    with _chat_processing_mutex:
        _chat_processing[chat_id] = processing

def should_cancel_chat(chat_id):
    """检查请求是否应该被取消"""
    with _chat_cancel_mutex:
        return _chat_should_cancel.get(chat_id, False)

def set_chat_cancel(chat_id, should_cancel):
    """设置请求取消标志"""
    with _chat_cancel_mutex:
        if should_cancel:
            _chat_should_cancel[chat_id] = True
        elif chat_id in _chat_should_cancel:
            del _chat_should_cancel[chat_id]

def is_duplicate_request(msg_id):
    """检查消息ID是否已处理过（去重）"""
    with _recent_msg_ids_mutex:
        if msg_id in _recent_msg_ids:
            return True
        # 记录这个消息ID
        _recent_msg_ids.append(msg_id)
        # 保持列表大小在限制内
        if len(_recent_msg_ids) > _MAX_RECENT_MSG_IDS:
            _recent_msg_ids.pop(0)
        return False

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
            delay = random.uniform(0.5, 2.0)
            time.sleep(delay)

def cleanup_old_memorys(max_age_hours=24):
    """清理过时的群聊和私聊记忆"""
    current_time = time.time()
    to_delete = []
    with _chat_processing_mutex:
        for chat_id, memory in list(memorys.items()):
            if memory.last_spoken_time:
                # 计算距离上次发言的时间（小时）
                time_diff = (current_time - memory.last_spoken_time.timestamp()) / 3600
                if time_diff > max_age_hours:
                    to_delete.append(chat_id)

        for chat_id in to_delete:
            del memorys[chat_id]
            # 清理对应的处理状态（可能是群聊或私聊）
            group_key = f"group_{chat_id}"
            private_key = f"private_{chat_id}"
            if group_key in _chat_processing:
                del _chat_processing[group_key]
            if private_key in _chat_processing:
                del _chat_processing[private_key]
            print(f"Cleaned up memory for chat {chat_id}")

def _process_event(event_data):
    """
    核心事件处理逻辑（被 / 和 /internal/process 共用）
    """
    logger = group_chat_logger if event_data.get('message_type') == 'group' else private_chat_logger
    target_groups = _server_cfg["target_groups"]
    target_users = _server_cfg["target_users"]
    global use_custom_prompt, memorys

    # 去重检查：防止重复处理同一条消息
    msg_id = event_data.get('message_id')
    print(f"[EVENT] type={event_data.get('message_type')}, id={msg_id}, from={event_data.get('user_id') or event_data.get('group_id')}")
    if msg_id and is_duplicate_request(msg_id):
        print(f"[SKIP] 重复请求已忽略: msg_id={msg_id}")
        logger.info(f"重复请求已忽略: msg_id={msg_id}")
        return jsonify({'status': 'ok', 'reason': 'duplicate'})
    # 调试日志
    if debug:
        print(f"[DEBUG] Received event: msg_type={event_data.get('message_type')}, msg_id={msg_id}, from={event_data.get('user_id') or event_data.get('group_id')}")

    cmd_result = command.handle_cmd(event_data)
    if cmd_result is not None:
        # 处理 Agent 返回的状态变更
        if cmd_result.state_changes.get("use_custom_prompt") is not None:
            use_custom_prompt = cmd_result.state_changes["use_custom_prompt"]
        return jsonify({
            'status': 'ok',
        })
    # 按聊天对象开启/关闭
    if event_data.get('raw_message') == CONFIG["chatbot_server"].get("trun_on_cmd"):
        if event_data.get('message_type') == 'private':
            chat_target = event_data.get('user_id')
        else:
            chat_target = event_data.get('group_id')
        if chat_target:
            with _disabled_chats_mutex:
                _disabled_chats.discard(chat_target)
            logger.info(f"Chatbot enabled for {chat_target}")
    if event_data.get('raw_message') == CONFIG["chatbot_server"].get("trun_off_cmd"):
        if event_data.get('message_type') == 'private':
            chat_target = event_data.get('user_id')
        else:
            chat_target = event_data.get('group_id')
        if chat_target:
            with _disabled_chats_mutex:
                _disabled_chats.add(chat_target)
            logger.info(f"Chatbot disabled for {chat_target}")

    # ——————————————————PRIVATE MESSAGE————————————————————
    if event_data.get('message_type') == 'private':
        if event_data.get('user_id') in target_users:
            target_user = event_data.get('user_id')
            is_enabled = target_user not in _disabled_chats
            if debug:
                print(f"[DEBUG] Private message from {target_user}, enabled={is_enabled}")
            if is_enabled:
                chat_id = f"private_{target_user}"

                # 私聊场景：如果正在处理中，设置取消标志并等待
                if is_chat_processing(chat_id):
                    print(f"私聊 {chat_id} 正在处理中，设置取消标志并等待新消息")
                    set_chat_cancel(chat_id, True)

                    # 等待一小段时间，让旧请求完成或被取消
                    max_wait = 2.0  # 最多等待2秒
                    wait_interval = 0.1
                    waited = 0
                    while is_chat_processing(chat_id) and waited < max_wait:
                        time.sleep(wait_interval)
                        waited += wait_interval

                    # 如果仍在处理，强制清除旧请求状态（可能卡住了）
                    if is_chat_processing(chat_id):
                        logger.warning(f"私聊 {chat_id} 等待超时，强制清除旧请求状态，处理新请求")
                        set_chat_processing(chat_id, False)
                        # 不拒绝，继续处理新请求

                # 标记为处理中，清除取消标志
                set_chat_processing(chat_id, True)
                set_chat_cancel(chat_id, False)

                try:
                    time.sleep(random.randint(2, 4))

                    # 处理前再次检查是否应该取消
                    if should_cancel_chat(chat_id):
                        logger.info(f"私聊 {chat_id} 请求被取消（有新消息）")
                        return jsonify({'status': 'cancelled'})

                    msg_models = db.get_latest_messages_by_time(target_user, False, 30)
                    has_master_message = False
                    for msg in msg_models:
                        if msg.is_master:
                            print("Master message detected, AI will not reply.")
                            has_master_message = True
                            break
                    if len(msg_models) < 20 :
                        msg_models = db.get_latest_messages_by_count(target_user,False,20)
                    if not has_master_message:
                        # 获取或创建私聊记忆
                        memory = memorys.get(target_user)
                        if not memory:
                            memory = PersonaMemory(target_user)
                            memorys[target_user] = memory

                        reply = AI_agent.call_AI_agent(logger, msg_models, memory, is_group=False, use_custom_prompt=use_custom_prompt, target_id=target_user)
                        if reply and not debug:
                        # 发送前再次检查是否应该取消
                            if should_cancel_chat(chat_id):
                                logger.info(f"私聊 {chat_id} 请求被取消（有新消息），不发送回复")
                                return jsonify({'status': 'cancelled'})
                                        # 检查是否有master消息
                            has_master_message = False

                            send_messages_with_delay(send_private_message, target_user, reply, logger)
                except Exception as e:
                    logger.error(f"Error processing private message: {e}")
                    print(f"Error processing private message: {e}")
                finally:
                    # 处理完成,清除状态
                    set_chat_processing(chat_id, False)
                    set_chat_cancel(chat_id, False)

    # ——————————————————GROUP MESSAGE————————————————————
    if event_data.get('message_type') == 'group' and event_data.get('group_id') in target_groups and event_data.get('user_id') != event_data.get('self_id') and event_data.get('group_id') not in _disabled_chats:
        target_group = event_data.get('group_id')
        chat_id = f"group_{target_group}"
        print("Received event:", event_data)
        # 检查是否正在处理中,如果是则直接拒绝
        if is_chat_processing(chat_id):
            print(f"拒绝请求: {chat_id} 正在处理中")
            return jsonify({
                'status': 'rejected',
                'reason': 'busy'
            })

        # 标记为处理中
        set_chat_processing(chat_id, True)
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
            
            if len(msg_models) > 1:
                print(msg_models[-2].to_str())
            
            # 检查是否有master消息
            has_master_message = False
            for msg in msg_models:
                if msg.is_master:
                    print("Master message detected, AI will not reply.")
                    has_master_message = True
                    break
            
            if not has_master_message:
                memory = memorys.get(target_group)
                if not memory:
                    memory = PersonaMemory(target_group)
                    memorys[target_group] = memory
                # reply,memory = group_pipline.handle_group_chat(logger,msg_models,memory)
                reply = AI_agent.call_AI_agent(logger,msg_models,memory,True,use_custom_prompt,target_id=target_group)
                if reply and not debug:
                    send_messages_with_delay(send_group_message, target_group, reply, logger)
        except Exception as e:
            logger.error(f"Error processing group message: {e}")
            print(f"Error processing group message: {e}")
        finally:
            # 处理完成,清除状态
            set_chat_processing(chat_id, False)

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
