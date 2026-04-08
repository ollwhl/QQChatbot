import httpx
from bs4 import BeautifulSoup
from urllib.parse import quote_plus

from config import GOOGLE_API_KEY, GOOGLE_CSE_ID, logger
from tools.browser import fetch_page


def _format_results(results: list[tuple[str, str, str]]) -> str:
    lines = []
    for i, (title, link, snippet) in enumerate(results, 1):
        lines.append(f"{i}. {title}\n   URL: {link}\n   摘要: {snippet}")
    return "\n\n".join(lines)


async def _search_via_api(query: str, num_results: int) -> str:
    """通过 Google Custom Search JSON API 搜索。"""
    url = "https://www.googleapis.com/customsearch/v1"
    params = {
        "key": GOOGLE_API_KEY,
        "cx": GOOGLE_CSE_ID,
        "q": query,
        "num": min(num_results, 10),
    }
    async with httpx.AsyncClient(timeout=15) as client:
        resp = await client.get(url, params=params)
        resp.raise_for_status()
        data = resp.json()

    items = data.get("items", [])
    if not items:
        return "未找到相关结果。"

    results = [
        (item.get("title", "无标题"), item.get("link", ""), item.get("snippet", ""))
        for item in items
    ]
    return _format_results(results)


async def _search_via_google(query: str, num_results: int) -> str | None:
    """用 Selenium 抓取 Google 搜索结果。"""
    search_url = f"https://www.google.com/search?q={quote_plus(query)}&num={num_results}&hl=zh-CN"
    try:
        html = await fetch_page(search_url, wait_seconds=5)
    except Exception as e:
        logger.warning("Google Selenium 抓取失败: %s", e)
        return None

    soup = BeautifulSoup(html, "html.parser")
    results: list[tuple[str, str, str]] = []

    # 策略1: div.g
    for g in soup.select("div.g"):
        anchor = g.select_one("a[href]")
        title_el = g.select_one("h3")
        if not anchor or not title_el:
            continue
        link = anchor["href"]
        if not link.startswith("http"):
            continue
        title = title_el.get_text(strip=True)
        snippet_el = g.select_one("div.VwiC3b") or g.select_one("span.st")
        snippet = snippet_el.get_text(strip=True) if snippet_el else ""
        results.append((title, link, snippet))

    # 策略2: div.tF2Cxc
    if not results:
        for div in soup.select("div.tF2Cxc"):
            a = div.find("a", href=True)
            h3 = div.find("h3")
            if not a or not h3:
                continue
            link = a["href"]
            if not link.startswith("http"):
                continue
            title = h3.get_text(strip=True)
            for tag in div.find_all(["h3", "cite"]):
                tag.decompose()
            snippet = div.get_text(separator=" ", strip=True)[:200]
            results.append((title, link, snippet))

    # 策略3: 通用 a > h3
    if not results:
        for h3 in soup.find_all("h3"):
            a = h3.find_parent("a")
            if not a or not a.has_attr("href"):
                continue
            link = a["href"]
            if not link.startswith("http"):
                continue
            title = h3.get_text(strip=True)
            results.append((title, link, ""))

    if not results:
        logger.warning("Google Selenium 未解析到结果 (HTML 长度=%d)", len(html))
        return None

    logger.info("Google Selenium 解析到 %d 条结果", len(results))
    return _format_results(results[:num_results])


async def _search_via_duckduckgo(query: str, num_results: int) -> str:
    """用 Selenium 抓取 DuckDuckGo 搜索结果（回退方案）。"""
    search_url = f"https://duckduckgo.com/?q={quote_plus(query)}&t=h_&ia=web"
    try:
        html = await fetch_page(search_url, wait_seconds=5)
    except Exception as e:
        return f"搜索请求失败: {e}"

    soup = BeautifulSoup(html, "html.parser")
    results: list[tuple[str, str, str]] = []

    # DuckDuckGo JS 渲染后的结构: article 标签或 li[data-layout] 内含搜索结果
    for article in soup.find_all("article"):
        # 找含外部链接的 a 标签（跳过 duckduckgo 自身链接）
        links = [
            a for a in article.find_all("a", href=True)
            if a["href"].startswith("http") and "duckduckgo.com" not in a["href"]
        ]
        if not links:
            continue
        # 通常第二个 a 是带标题文本的结果链接（第一个是 URL 显示）
        best_a = None
        for a in links:
            text = a.get_text(strip=True)
            if len(text) > 5 and not text.startswith("http"):
                best_a = a
                break
        if not best_a:
            best_a = links[0]
        title = best_a.get_text(strip=True)
        href = best_a["href"]
        # 摘要: article 内最长的 span 文本（排除标题和 URL）
        spans = [
            s.get_text(strip=True) for s in article.find_all("span")
            if len(s.get_text(strip=True)) > 30 and s.get_text(strip=True) != title
        ]
        snippet = spans[0][:200] if spans else ""
        results.append((title, href, snippet))
        if len(results) >= num_results:
            break

    # 回退: HTML 版 DuckDuckGo 结构 (html.duckduckgo.com/html/)
    if not results:
        for r in soup.select("div.result"):
            if "result--ad" in r.get("class", []):
                continue
            a = r.select_one("a.result__a")
            if not a:
                continue
            title = a.get_text(strip=True)
            href = a.get("href", "")
            if not href.startswith("http") or "duckduckgo.com/y.js" in href:
                continue
            snippet_el = r.select_one("a.result__snippet")
            snippet = snippet_el.get_text(strip=True) if snippet_el else ""
            results.append((title, href, snippet))
            if len(results) >= num_results:
                break

    if not results:
        logger.warning("DuckDuckGo Selenium 未解析到结果 (HTML 长度=%d)", len(html))
        return "未找到相关结果。"

    return _format_results(results)


async def google_search(query: str, num_results: int = 5) -> str:
    """搜索互联网，返回标题+URL+摘要列表。优先级: Google API → Google Selenium → DuckDuckGo Selenium。"""
    # 1. Google Custom Search API
    if GOOGLE_API_KEY and GOOGLE_CSE_ID:
        try:
            result = await _search_via_api(query, num_results)
            logger.info("搜索通过 Google API 完成")
            return result
        except Exception as e:
            logger.warning("Google API 搜索失败: %s", e)

    # 2. Google Selenium
    result = await _search_via_google(query, num_results)
    if result:
        logger.info("搜索通过 Google Selenium 完成")
        return result

    # 3. DuckDuckGo Selenium 回退
    logger.info("Google 不可用，回退到 DuckDuckGo Selenium")
    return await _search_via_duckduckgo(query, num_results)
