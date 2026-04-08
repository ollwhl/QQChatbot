import logging
import logging.handlers
import traceback
import os
from datetime import datetime

# 日志目录：项目根目录下的 log/
_log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'log')
os.makedirs(_log_dir, exist_ok=True)

def _setup_logger(name, log_file, level=logging.INFO):
    """通用日志记录器设置"""
    # 防止重复添加 handlers
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(level)
    
    # 创建一个文件 handler，并设置级别
    # 使用 TimedRotatingFileHandler 来按天轮换日志
    handler = logging.handlers.TimedRotatingFileHandler(
        log_file, when='midnight', interval=1, backupCount=7, encoding='utf-8'
    )
    handler.setLevel(level)
    
    # 创建一个 formatter
    # 新的格式化器将只输出消息本身，因为所有上下文都将包含在消息中
    formatter = logging.Formatter('%(message)s')
    
    # 设置 formatter
    handler.setFormatter(formatter)
    
    # 添加 handler
    logger.addHandler(handler)
    
    return logger

# 创建两个独立的 logger 实例
group_chat_logger = _setup_logger('group_chat', os.path.join(_log_dir, 'group_chat_ai.log'))
private_chat_logger = _setup_logger('private_chat', os.path.join(_log_dir, 'private_chat_ai.log'))

def _format_log_message(func_name: str, system_prompt: str, user_prompt: str, response: str, status: str, exception_info: str = "") -> str:
    """格式化日志消息"""
    dt = datetime.now().strftime("%Y/%m/%d/%H:%M:%S")
    log_entry = [
        "==================== AI Interaction Start ====================",
        f"Time: {dt}",
        f"Function: {func_name}",
        "----- System Prompt -----",
        system_prompt,
        "----- User Prompt -----",
        user_prompt,
        "----- AI Response -----",
        str(response),
        f"----- Status: {status} -----"
    ]
    if exception_info:
        log_entry.extend([
            "----- Exception -----",
            exception_info
        ])
    log_entry.append("===================== AI Interaction End =====================\n")
    return "\n".join(log_entry)

def log_ai_interaction(logger, func_name: str, system_prompt: str, user_prompt: str, response: str, status: str, exc = None):
    """
    记录一次完整的 AI 交互。

    """
    exception_info = None
    if exc:
        # 捕获完整的堆栈跟踪信息
        exception_info = ''.join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    
    log_message = _format_log_message(func_name, system_prompt, user_prompt, response, status, exception_info) # type: ignore
    print(log_message)
    if status == 'FAILURE':
        logger.error(log_message)
    else:
        logger.info(log_message)

if __name__ == '__main__':
    # 这是一个如何使用的例子
    # logger.py
    
    # 模拟一次成功的群聊交互
    log_ai_interaction(
        logger=group_chat_logger,
        func_name="msg_manger",
        system_prompt="You are a helpful assistant.",
        user_prompt="Hello, who are you?",
        response='{"should_reply": true}',
        status="SUCCESS"
    )

    # 模拟一次失败的私聊交互
    try:
        raise ValueError("This is a test error")
    except ValueError as e:
        log_ai_interaction(
            logger=private_chat_logger,
            func_name="generate_chat_response",
            system_prompt="You are a poet.",
            user_prompt="Write a poem about the sea.",
            response="(No reply generated)",
            status="FAILURE",
            exc=e
        )
