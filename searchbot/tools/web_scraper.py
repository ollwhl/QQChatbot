from bs4 import BeautifulSoup

from config import MAX_CONTENT_LENGTH, logger
from tools.browser import fetch_page

_NOISE_TAGS = [
    "script", "style", "nav", "footer", "header",
    "aside", "noscript", "iframe", "svg",
]


async def scrape_url(url: str) -> str:
    """用 Selenium 爬取指定 URL 的网页正文内容。"""
    try:
        html = await fetch_page(url)
    except Exception as e:
        logger.error("scrape_url 失败: %s, 错误: %s", url, e)
        return f"无法访问 {url}: {e}"

    soup = BeautifulSoup(html, "html.parser")

    for tag in soup(_NOISE_TAGS):
        tag.decompose()

    title = soup.title.string.strip() if soup.title and soup.title.string else ""
    text = soup.get_text(separator="\n", strip=True)

    if len(text) > MAX_CONTENT_LENGTH:
        text = text[:MAX_CONTENT_LENGTH] + "\n...(内容已截断)"

    result = f"标题: {title}\nURL: {url}\n\n{text}" if title else f"URL: {url}\n\n{text}"
    return result
