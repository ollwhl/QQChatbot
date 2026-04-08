import requests
import json
from urllib.parse import urljoin, urlparse
import re
import html
from datetime import datetime
import time
import random
import time
from call_llm import describe_image
from database import db, MessageModel
from config import CONFIG

_server_cfg = CONFIG["chatbot_server"]
base_url = _server_cfg["NAPCAT_HOST"]
token = _server_cfg["NAPCAT_TOKEN"]
_master_user_id = CONFIG["master_user_id"]
def send_ai_identify(msg_id):
    updated_count = 0
    max_retries = 3
    updated_count = db.update_is_ai(msg_id,True)
    if updated_count != 0:
        print(f"{msg_id} update is_ai success")
        return True
    print(f"{msg_id} update is_ai failed tried")
    return False


def send_private_message(user_id,msg):
    if msg is None:
        return
    url = urljoin(base_url,"send_private_msg")
    payload = json.dumps({
    "user_id": user_id,
    "message": [
        {
            "type": "text",
            "data": {
                "text": msg
            }
        }
    ]
    })
    headers = {
    'Content-Type': 'application/json'
    }
    headers['Authorization'] = f'Bearer {token}'
    response = requests.request("POST", url, headers=headers, data=payload)
    resp_data = json.loads(response.text)
    if resp_data.get("status") == "ok":
        msg_id = resp_data.get("data", {}).get("message_id")
        # 直接将机器人发送的消息插入数据库，设置 is_ai=True 和正确的 peer_id
        bot_msg = MessageModel(
            msg_id=msg_id,
            sender_id=_master_user_id,
            group_id=0,
            sender_name="chatbot",
            sender_card="",
            content=msg,
            timestamp=datetime.now().strftime("%Y%m%d%H%M%S"),
            is_ai=True,
            is_group=False,
            peer_id=user_id
        )
        db.add_msg(bot_msg)
        send_ai_identify(msg_id)
    return response.text

def get_private_message(user_id,msg_length):
    url = urljoin(base_url,"/get_friend_msg_history")
    payload = json.dumps({
        "user_id": user_id,
        "message_seq": 0,
        "count": msg_length,
        "reverseOrder": False
    })
    headers = {
        'Content-Type': 'application/json'
    }
    headers['Authorization'] = f'Bearer {token}'
    response = requests.request("POST", url, headers=headers, data=payload)
    try:
        response_data = json.loads(response.text)  

    except json.JSONDecodeError as e:
        print(f"JSON解析失败: {e}")
        return []  
    
    conversation_list = transform_napcat_response(response_data)
    return conversation_list

def send_group_message(group_id,msg):
    if msg is None:
        return
    url = urljoin(base_url,"send_group_msg")
    payload = json.dumps({
    "group_id": group_id,
    "message": [
        {
            "type": "text",
            "data": {
                "text": msg
            }
        }
    ]
    })
    headers = {
    'Content-Type': 'application/json'
    }
    headers['Authorization'] = f'Bearer {token}'
    response = requests.request("POST", url, headers=headers, data=payload)
    resp_data = json.loads(response.text)
    if resp_data.get("status") == "ok":
        msg_id = resp_data.get("data", {}).get("message_id")
        # 直接将机器人发送的消息插入数据库，设置 is_ai=True
        bot_msg = MessageModel(
            msg_id=msg_id,
            sender_id=_master_user_id,
            group_id=group_id,
            sender_name="bot",
            sender_card="",
            content=msg,
            timestamp=datetime.now().strftime("%Y%m%d%H%M%S"),
            is_ai=True,
            is_group=True
        )
        db.add_msg(bot_msg)
    return response.text
    
def get_group_message(group_id,msg_length=20):
    url = urljoin(base_url,"/get_group_msg_history")
    payload = json.dumps({
        "group_id": group_id,
        "message_seq": 0,
        "count": msg_length,
        "reverseOrder": False
    })
    headers = {
        'Content-Type': 'application/json'
    }
    headers['Authorization'] = f'Bearer {token}'
    response = requests.request("POST", url, headers=headers, data=payload)
    try:
        response_data = json.loads(response.text)  

    except json.JSONDecodeError as e:
        print(f"JSON解析失败: {e}")
        return []  
    # print ("response_data:",response_data)
    # conversation_list = transform_napcat_response(response_data)
    return response_data

def transform_napcat_response(api_response, time_format="%Y%m%d%H%M%S"):
    """
    将NapCat获取消息<的API响应，转换为对话列表。
    增强1：自动识别输入是字典（原始响应）还是列表（已转换）。
    增强2：解析CQ码，使内容更易读。
    """
    result = []
    
    if not isinstance(api_response, dict):
        print("错误：transform_napcat_response 输入不是字典也不是列表")
        return result
    
    if api_response.get('retcode') != 0:
        print(f"API调用失败：{api_response.get('message', '未知错误')},retcode:{api_response.get('retcode')}")
        return result
    
    messages = api_response.get('data', {}).get('messages', [])
    if not messages:
        print("提示：消息列表为空")
        return result
    
    for msg in messages:
        msg_parsed =  parse_msg(msg,time_format)
        result.append(msg_parsed)
    return result
def get_stranger_info(user_id):
    url = urljoin(base_url,"/get_stranger_info")
    payload = json.dumps({
    "user_id": user_id,
    "no_cache": False
    })
    headers = {
    'Content-Type': 'application/json'
    }
    headers['Authorization'] = f'Bearer {token}'
    response = requests.request("POST", url, headers=headers, data=payload)
    return response.text
def get_group_member_info(group_id, user_id):
    url = urljoin(base_url,"/get_group_member_info")
    payload = json.dumps({
    "group_id": group_id,
    "user_id": user_id,
    "no_cache": False
    })
    headers = {
    'Content-Type': 'application/json'
    }
    headers['Authorization'] = f'Bearer {token}'
    response = requests.request("POST", url, headers=headers, data=payload)
    return response.text
def get_msg_by_id(id):
    url = urljoin(base_url,"/get_msg")
    payload = json.dumps({
    "message_id": id
    })
    headers = {
    'Content-Type': 'application/json'
    }
    headers['Authorization'] = f'Bearer {token}'
    response = requests.request("POST", url, headers=headers, data=payload)
    return response.text

def parse_msg(msg, time_format="%Y%m%d%H%M%S", quick_mode=False) -> MessageModel:
        # %d: 0埋めした10進数で表記した月中の日にち
        # %m: 0埋めした10進数で表記した月
        # %y: 0埋めした10進数で表記した西暦の下2桁
        # %Y: 0埋めした10進数で表記した西暦4桁
        # %H: 0埋めした10進数で表記した時 （24時間表記）
        # %I: 0埋めした10進数で表記した時 （12時間表記）
        # %M: 0埋めした10進数で表記した分
        # %S: 0埋めした10進数で表記した秒
        # %f: 0埋めした10進数で表記したマイクロ秒（6桁）
        # %A: ロケールの曜日名
        # %a: ロケールの曜日名（短縮形）
        # %B: ロケールの月名
        # %b: ロケールの月名（短縮形）
        # %j: 0埋めした10進数で表記した年中の日にち（正月が'001'）
        # %U: 0埋めした10進数で表記した年中の週番号 （週の始まりは日曜日）
        # %W: 0埋めした10進数で表記した年中の週番号 （週の始まりは月曜日）
        def sender_info(sender_dict):
            if sender_dict.get('id') == _master_user_id:
                return f"{CONFIG['master_name']}(我)"
            if not sender_dict.get('card'):
                return sender_dict.get('nickname', '未知用户')
            else:
                return f"{sender_dict.get('card', '未知用户')}({sender_dict.get('nickname', '未知用户')})"
        #print("原始消息内容:", msg)
        sender_dict = msg.get('sender', {})
        sender = sender_info(sender_dict)
        # 处理时间戳
        time_str = ""
        timestamp = msg.get('time', 0)
        try:
            dt = datetime.fromtimestamp(timestamp)
            time_str = dt.strftime(time_format)
        except:
            time_str = "未知时间"
        
        # 处理 message 字段
        message_content = msg.get('message')
        full_content = ""
        
        if isinstance(message_content, str):
            full_content = message_content
        elif isinstance(message_content, list):
            content_parts = []
            for msg_segment in message_content:
                if isinstance(msg_segment, dict):
                    msg_type = msg_segment.get('type')
                    msg_data = msg_segment.get('data', {})
                    if msg_type == 'text':
                        content_parts.append(msg_data.get('text', ''))
                    elif msg_type == 'at':
                        qq_id = msg_data.get('qq', '')
                        #print(get_group_member_info(msg.get('group_id',0),qq_id))
                        at_user_name = ""
                        at_user_info= json.loads(get_group_member_info(msg.get('group_id',0),qq_id)).get('data',{})
                        if at_user_info.get('user_id') == _master_user_id:
                            at_user_name = f"{CONFIG['master_name']}(我)"
                        elif not at_user_info.get('card'):
                            at_user_name = at_user_info.get('nickname', '未知用户')
                        else:
                            at_user_name = at_user_info.get('card', '未知用户')
                        display = f"@{at_user_name}"
                        content_parts.append(display)
                    elif msg_type == 'image':
                        image_url = msg_data.get('url', '')
                        image_summary = msg_data.get('summary', '')
                        if image_summary:
                            # QQ 已有描述，直接使用
                            content_parts.append(f'[图片：{image_summary}]')
                        elif image_url and not quick_mode:
                            # 非快速模式：调用 AI 解析图片
                            description = describe_image(image_url)
                            content_parts.append(f'[图片：{description}]')
                        else:
                            # 快速模式或无URL：暂时标记为[图片]
                            content_parts.append('[图片]')
                    elif msg_type == 'face':
                        content_parts.append(f'[表情：{msg_data.get("id", "未知")}]')
                    elif msg_type == 'reply':
                        reply_id = msg_data.get('id', '')
                        reply_msg = json.loads(get_msg_by_id(reply_id)).get('data',{})
                        #print("回复的消息内容:", get_msg_by_id(reply_id))
                        reply_msg_sender = sender_info(reply_msg.get('sender', {}))
                        reply_msg_content = reply_msg.get('raw_message', '')

                        content_parts.append(f'[回复消息：{reply_msg_sender}: {reply_msg_content}]')
                    else:
                        content_parts.append(f'[{msg_type}类型消息，无法解析，请无视]')
                elif isinstance(msg_segment, str):
                    content_parts.append(msg_segment)
            full_content = ''.join(content_parts)
        elif message_content is not None:
            full_content = str(message_content)
        
        # ====== 新增：关键步骤，解析CQ码 ======
        full_content = parse_cq_code(full_content)
        is_group = msg.get('message_type') == "group"
        # 私聊消息设置 peer_id 为发送者（对方）的 user_id
        peer_id = None if is_group else sender_dict.get('user_id')
        if full_content.strip():
            return MessageModel(
                msg_id=msg.get('message_id'),
                sender_id=sender_dict.get('user_id'),
                group_id=msg.get('group_id'),
                sender_name=sender_dict.get('nickname'),
                sender_card=sender_dict.get('card'),
                content=full_content,
                timestamp=time_str,
                is_ai=False,
                is_group=is_group,
                peer_id=peer_id
            )
        return None # type: ignore


    

def extract_image_urls(msg) -> list:
    """
    从消息中提取所有图片URL
    返回: [(image_url, has_summary), ...]
    """
    message_content = msg.get('message')
    image_urls = []

    if isinstance(message_content, list):
        for msg_segment in message_content:
            if isinstance(msg_segment, dict):
                msg_type = msg_segment.get('type')
                if msg_type == 'image':
                    msg_data = msg_segment.get('data', {})
                    image_url = msg_data.get('url', '')
                    image_summary = msg_data.get('summary', '')
                    # 只收集没有summary且有URL的图片
                    if image_url and not image_summary:
                        image_urls.append(image_url)

    return image_urls

def async_update_image_descriptions(msg_id, image_urls, current_content):
    """
    后台线程：解析图片并更新数据库

    Args:
        msg_id: 消息ID
        image_urls: 图片URL列表
        current_content: 当前消息内容（包含[图片]占位符）
    """
    import threading

    def _update():
        try:
            from database import db

            # 解析所有图片
            new_content = current_content
            for image_url in image_urls:
                desc = describe_image(image_url)
                # 只替换第一个 [图片]
                new_content = new_content.replace('[图片]', f'[图片：{desc}]', 1)

            # 如果内容有变化，更新数据库
            if new_content != current_content:
                db.update_message_content(msg_id, new_content)
                print(f"异步图片解析完成: msg_id={msg_id}")

        except Exception as e:
            print(f"异步图片解析失败: {e}")

    # 启动后台线程
    threading.Thread(target=_update, daemon=True).start()

def parse_cq_code(content: str) -> str:
    """
    将消息内容中的CQ码字符串替换为更易读的格式。
    """
    if not isinstance(content, str):
        return content
    
    # 1. 处理图片 [CQ:image,...]
    def replace_image(match):
        # 可以进一步提取文件名，这里统一替换为[图片]
        # 例如：file=206F8EB9... -> [图片:206F8EB9...]
        return '[图片]'
    content = re.sub(r'\[CQ:image[^\]]*\]', replace_image, content)
    
    # 2. 处理回复 [CQ:reply,id=123456]
    content = re.sub(r'\[CQ:reply,id=(\d+)\]', r'[回复消息\1]', content)
    
    # 3. 处理@某人 [CQ:at,qq=123456]
    content = re.sub(r'\[CQ:at,qq=(\d+)\]', r'@\1', content)
    
    # 4. 处理表情 [CQ:face,id=123] (你的数据里没有，但可能遇到)
    content = re.sub(r'\[CQ:face,id=(\d+)\]', r'[表情\1]', content)
    
    # 5. 处理JSON小程序等复杂CQ码 [CQ:json,...]
    # 简单替换为[小程序]，也可尝试解析json提取标题（见下文注释）
    content = re.sub(r'\[CQ:json[^\]]*\]', '[QQ小程序]', content)
    
    # 6. 清理HTML实体（你输出中的&#91;代表[，&#93;代表]）
    content = html.unescape(content)
    
    return content

if __name__ == '__main__':
    # print(send_private_message(1395139096,"给大鹤打钱")) #小伞
    #msg_dicts = get_private_message (1395139096,3) #叶纸
    #print(msg_dicts)
    # msg_dicts = get_group_message(1045042739,5)
    # msgs = []
    # for item in msg_dicts:
    #     sender = f"{item['sender_card']}({item['sender_name']})"
    #     content = item.get("content", "")
    #     msg_time = item.get("timestamp", "")
    #     if msg_time:
    #         msgs.append(f"[{msg_time}] {sender}: {content}")
    #     else:
    #         msgs.append(f"{sender}: {content}")
    # print("\n".join(msgs))
    # with open("msg.txt","w") as f:
    #     f.write("\n".join(msgs))
    # print(get_msg_by_id(199273306))
    # print(send_group_message(1035078631,input("输入群消息内容:")))
    # print(db.update_is_ai(1839050409,True))
    #print(get_group_member_info(1045042739,2727873726))
    # send_private_message(1395139096,"test")
    msgs = db.get_latest_messages_by_count(1395139096,False,20)
    for msg in msgs :print(msg.to_str(),msg.is_ai,msg.is_master) 
