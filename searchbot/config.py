import logging
import os
from dotenv import load_dotenv
from datetime import datetime
load_dotenv()

LOG_LEVEL: str = os.getenv("LOG_LEVEL", "DEBUG")
LOG_FILE: str = os.getenv("LOG_FILE", "searchbot.log")


def setup_logger() -> logging.Logger:
    logger = logging.getLogger("searchbot")
    logger.setLevel(getattr(logging, LOG_LEVEL.upper(), logging.INFO))
    logger.handlers.clear()

    fmt = logging.Formatter(
        "[%(asctime)s] %(levelname)s %(name)s.%(funcName)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    fh = logging.FileHandler(LOG_FILE, encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    # 只在 DEBUG 级别同时输出到终端
    if LOG_LEVEL.upper() == "DEBUG":
        sh = logging.StreamHandler()
        sh.setFormatter(fmt)
        logger.addHandler(sh)

    return logger


logger = setup_logger()

OPENAI_API_KEY: str = os.getenv("DEEPSEEK_API_KEY", "")
OPENAI_BASE_URL: str = "https://api.deepseek.com"
MODEL_NAME: str = "deepseek-chat"

GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", "")
GOOGLE_CSE_ID: str = os.getenv("GOOGLE_CSE_ID", "")

MAX_CONTENT_LENGTH: int = 8000
MAX_DEEP_EXPLORE_PAGES: int = 5
MAX_DEEP_EXPLORE_DEPTH: int = 2
time = datetime.now().strftime("%Y:%m:%d:%H:%M:%S")

SYSTEM_PROMPT: str = f"""你是一个联网 AI 助手，能够搜索互联网、爬取网页内容并深度探索网站。
请根据用户的问题，合理使用工具来获取信息并给出准确、详细的回答。
当前时间：{time}
**工具使用优先级规则：**
- 当用户的输入中包含明确的 URL（如 http://、https:// 开头的链接）时，必须优先使用 scrape_url 工具直接爬取该网页内容，而不是先用 google_search 去搜索。
- 如果用户提供了 URL 并且还附带了问题，且问题需要深入了解该网站的多个页面，则使用 deep_explore 工具。
- 只有当用户没有提供任何 URL，需要你主动查找信息时，才使用 google_search 工具进行搜索。

回答时请使用中文，除非用户要求使用其他语言。"""
