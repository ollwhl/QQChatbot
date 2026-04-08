import asyncio
from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup
from openai import AsyncOpenAI

from config import (
    MAX_CONTENT_LENGTH,
    MAX_DEEP_EXPLORE_PAGES,
    OPENAI_API_KEY,
    OPENAI_BASE_URL,
    MODEL_NAME,
    logger,
)
from tools.browser import fetch_page
from tools.web_scraper import scrape_url


def _extract_links(html: str, base_url: str) -> list[str]:
    """从 HTML 中提取同域链接。"""
    soup = BeautifulSoup(html, "html.parser")
    base_domain = urlparse(base_url).netloc
    links: list[str] = []
    seen: set[str] = set()

    for a in soup.find_all("a", href=True):
        href = urljoin(base_url, a["href"])
        parsed = urlparse(href)
        clean = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
        if (
            parsed.scheme in ("http", "https")
            and parsed.netloc == base_domain
            and clean not in seen
            and clean != base_url.rstrip("/")
        ):
            seen.add(clean)
            links.append(clean)

    return links


async def deep_explore(url: str, question: str) -> str:
    """深度探索：爬取目标页面及其子页面，汇总回答用户问题。"""
    logger.info("deep_explore 开始: url=%s, question=%s", url, question)

    # 1. 爬取主页面
    try:
        main_html = await fetch_page(url)
        logger.info("主页面 HTML 获取成功, 长度=%d", len(main_html))
    except Exception as e:
        logger.error("主页面爬取失败: %s", e)
        return f"无法访问 {url}: {e}"

    main_content = await scrape_url(url)
    logger.debug("主页面正文提取完成, 长度=%d", len(main_content))

    # 2. 提取相关链接
    all_links = _extract_links(main_html, url)
    logger.info("从主页面提取到 %d 个同域链接", len(all_links))

    # 用 LLM 筛选与问题最相关的链接
    client = AsyncOpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)
    if all_links:
        link_list = "\n".join(f"- {l}" for l in all_links[:30])
        filter_prompt = (
            f"以下是从 {url} 提取的链接列表：\n{link_list}\n\n"
            f"用户的问题是：{question}\n\n"
            f"请从中选出最多 {MAX_DEEP_EXPLORE_PAGES} 个与问题最相关的链接，"
            f"每行一个 URL，不要输出其他内容。如果没有相关链接，输出「无」。"
        )
        logger.info("发送链接筛选请求给 LLM")
        logger.debug("链接筛选 prompt: %s", filter_prompt)
        filter_resp = await client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": filter_prompt}],
            temperature=0,
        )
        selected_text = filter_resp.choices[0].message.content or ""
        logger.info("LLM 链接筛选响应: %s", selected_text)
        selected_links = [
            line.strip().lstrip("- ")
            for line in selected_text.strip().splitlines()
            if line.strip().startswith("http")
        ]
        logger.info("筛选出 %d 个相关链接: %s", len(selected_links), selected_links)
    else:
        selected_links = []

    # 3. 并发爬取子页面
    sub_contents: list[str] = []
    if selected_links:
        logger.info("开始并发爬取 %d 个子页面", len(selected_links[:MAX_DEEP_EXPLORE_PAGES]))
        tasks = [scrape_url(link) for link in selected_links[:MAX_DEEP_EXPLORE_PAGES]]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for link, result in zip(selected_links, results):
            if isinstance(result, Exception):
                logger.warning("子页面爬取失败: %s, 错误: %s", link, result)
                sub_contents.append(f"[{link}] 爬取失败: {result}")
            else:
                logger.debug("子页面爬取成功: %s, 长度=%d", link, len(result))
                sub_contents.append(result)

    # 4. 汇总
    combined = f"=== 主页面 ===\n{main_content}\n\n"
    for i, content in enumerate(sub_contents, 1):
        truncated = content[:MAX_CONTENT_LENGTH] if len(content) > MAX_CONTENT_LENGTH else content
        combined += f"=== 子页面 {i} ===\n{truncated}\n\n"

    # 截断总内容防止超出上下文
    if len(combined) > MAX_CONTENT_LENGTH * 3:
        combined = combined[: MAX_CONTENT_LENGTH * 3] + "\n...(内容已截断)"

    summary_prompt = (
        f"基于以下爬取的网页内容，请针对用户的问题进行全面、详细的回答。\n\n"
        f"用户问题：{question}\n\n"
        f"爬取内容：\n{combined}"
    )
    logger.info("发送汇总请求给 LLM, 内容长度=%d", len(summary_prompt))
    logger.debug("汇总 prompt: %.2000s", summary_prompt)
    summary_resp = await client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": summary_prompt}],
        temperature=0.3,
    )

    summary = summary_resp.choices[0].message.content or "深度探索未能生成摘要。"
    logger.info("deep_explore 完成, 摘要长度=%d", len(summary))
    logger.debug("摘要内容: %.1000s", summary)
    return summary
