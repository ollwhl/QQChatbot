from config import CONFIG
from cmd_agent import CmdAgent, CmdResult
import qq_msg

_agent = CmdAgent()


def handle_cmd(event_data: dict, session=None):
    """
    检测消息是否为命令（以 / 开头），如果是则交给 CmdAgent 处理。

    Returns:
        CmdResult | None: 命令执行结果，非命令返回 None
    """
    if not event_data:
        return None

    message_type = event_data.get('message_type')
    message = event_data.get('message', [])

    # 提取命令文本和上下文
    cmd_text = None
    context = {}

    if message_type == 'private':
        if message and isinstance(message, list) and len(message) > 0:
            first_segment = message[0]
            if first_segment.get('type') == 'text':
                raw_message = first_segment.get('data', {}).get('text', '')
                if raw_message.startswith("/"):
                    cmd_text = qq_msg.parse_msg(event_data).context.lstrip("/").strip()
                    context = {
                        "user_id": event_data.get('user_id'),
                        "is_group": False,
                        "session": session,
                    }

    elif message_type == 'group':
        if message and isinstance(message, list) and len(message) > 1:
            first_segment = message[0]
            second_segment = message[1]
            if (first_segment.get('type') == 'at' and
                str(first_segment.get('data', {}).get('qq')) == str(CONFIG["master_user_id"])):
                if second_segment.get('type') == 'text':
                    raw_message = second_segment.get('data', {}).get('text', '').strip()
                    if raw_message.startswith("/"):
                        cmd_text = qq_msg.parse_msg(event_data).context.lstrip("/").strip()
                        context = {
                            "group_id": event_data.get('group_id'),
                            "is_group": True,
                            "session": session,
                        }

    if cmd_text is None:
        return None

    # 交给 Agent 处理
    print(f"[Command] 收到命令: {cmd_text}")
    result = _agent.run(cmd_text, context)

    # 发送结果消息给用户
    if result.message:
        group_id = context.get("group_id")
        user_id = context.get("user_id")
        if group_id:
            qq_msg.send_group_message(group_id, result.message)
        elif user_id:
            qq_msg.send_private_message(user_id, result.message)

    return result
