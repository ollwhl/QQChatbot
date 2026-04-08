"""共享 Selenium 浏览器实例管理。

全局维护一个 headless Chrome 实例，避免每次请求都启动浏览器。
所有网页获取操作都通过 fetch_page() 进行。
"""

import asyncio
import atexit
import os
import signal
import time
from functools import partial
from pathlib import Path

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.common.exceptions import TimeoutException, WebDriverException

from config import logger

_PID_FILE = Path(__file__).resolve().parent.parent / ".chromedriver.pid"

_driver: webdriver.Chrome | None = None
_lock = asyncio.Lock()


def _kill_stale_processes() -> None:
    """清理上次运行残留的 chromedriver / chrome 进程。"""
    if not _PID_FILE.exists():
        return
    try:
        pids = _PID_FILE.read_text().strip().splitlines()
        for raw in pids:
            pid = int(raw.strip())
            try:
                os.kill(pid, signal.SIGTERM)
                logger.info("已终止残留进程: PID %d", pid)
            except ProcessLookupError:
                pass  # 进程已不存在
            except PermissionError:
                logger.warning("无权终止进程: PID %d", pid)
    except (ValueError, OSError) as e:
        logger.warning("读取 PID 文件失败: %s", e)
    finally:
        _PID_FILE.unlink(missing_ok=True)


def _save_pids(driver: webdriver.Chrome) -> None:
    """将 chromedriver 及其管理的 chrome 子进程 PID 写入文件。"""
    pids: list[str] = []
    # chromedriver 自身的 PID
    service_pid = getattr(driver.service, "process", None)
    if service_pid is not None:
        pids.append(str(service_pid.pid))
    # chrome 浏览器进程的 PID（通过 /proc 查找 chromedriver 的子进程）
    if service_pid is not None:
        try:
            children_dir = Path(f"/proc/{service_pid.pid}/task")
            if children_dir.exists():
                # 遍历 /proc 找父进程是 chromedriver 的进程
                for entry in Path("/proc").iterdir():
                    if not entry.name.isdigit():
                        continue
                    try:
                        stat = (entry / "stat").read_text()
                        ppid = stat.split(")")[1].split()[1]
                        if ppid == str(service_pid.pid):
                            pids.append(entry.name)
                    except (OSError, IndexError):
                        continue
        except OSError:
            pass
    if pids:
        _PID_FILE.write_text("\n".join(pids) + "\n")
        logger.debug("已记录进程 PID: %s", pids)


def _create_driver() -> webdriver.Chrome:
    _kill_stale_processes()

    opts = Options()
    opts.add_argument("--headless=new")
    opts.add_argument("--no-sandbox")
    opts.add_argument("--disable-dev-shm-usage")
    opts.add_argument("--disable-gpu")
    opts.add_argument("--disable-blink-features=AutomationControlled")
    opts.add_argument("--window-size=1920,1080")
    opts.add_argument("--lang=zh-CN")
    opts.add_argument(
        "--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/125.0.0.0 Safari/537.36"
    )
    # 隐藏 webdriver 特征
    opts.add_experimental_option("excludeSwitches", ["enable-automation"])
    opts.add_experimental_option("useAutomationExtension", False)
    # 使用 eager 策略：DOM 就绪即返回，不等所有资源加载完
    opts.page_load_strategy = "eager"

    driver = webdriver.Chrome(options=opts)
    driver.execute_cdp_cmd(
        "Page.addScriptToEvaluateOnNewDocument",
        {"source": "Object.defineProperty(navigator, 'webdriver', {get: () => undefined})"},
    )
    driver.set_page_load_timeout(60)
    driver.set_script_timeout(30)
    _save_pids(driver)
    logger.info("Chrome 浏览器实例已创建 (page_load_strategy=eager)")
    return driver


def _get_driver() -> webdriver.Chrome:
    global _driver
    if _driver is None:
        _driver = _create_driver()
    return _driver


def _reset_driver() -> webdriver.Chrome:
    """浏览器异常时重建实例。"""
    global _driver
    if _driver is not None:
        try:
            _driver.quit()
        except Exception:
            pass
        _driver = None
    return _get_driver()


def shutdown_browser() -> None:
    global _driver
    if _driver is not None:
        try:
            _driver.quit()
            logger.info("Chrome 浏览器已关闭")
        except Exception:
            pass
        _driver = None
    _PID_FILE.unlink(missing_ok=True)


atexit.register(shutdown_browser)


async def fetch_page(url: str, wait_seconds: float = 5) -> str:
    """用 Selenium 获取页面渲染后的 HTML。线程安全，支持异步调用。"""
    async with _lock:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, partial(_fetch_sync, url, wait_seconds))


def _fetch_sync(url: str, wait_seconds: float) -> str:
    driver = _get_driver()
    logger.info("Selenium 请求: %s", url)

    try:
        driver.get(url)
    except TimeoutException:
        # eager 策略下超时说明页面确实很慢，但 DOM 可能已部分可用
        logger.warning("driver.get() 超时，尝试获取已加载内容: %s", url)
    except WebDriverException as e:
        logger.error("Selenium 导航失败，重建浏览器: %s", e)
        driver = _reset_driver()
        driver.get(url)

    # 等待 body 出现，确保至少有基本 DOM
    try:
        WebDriverWait(driver, wait_seconds).until(
            lambda d: d.find_element(By.TAG_NAME, "body")
        )
    except TimeoutException:
        logger.warning("等待 body 超时: %s", url)

    # 额外等待一小段时间让 JS 渲染完成
    time.sleep(min(wait_seconds, 3))

    try:
        html = driver.page_source
    except WebDriverException as e:
        logger.error("获取 page_source 失败，重建浏览器: %s", e)
        driver = _reset_driver()
        raise RuntimeError(f"无法获取页面内容: {url}") from e

    logger.debug("Selenium 获取成功, HTML 长度=%d", len(html))
    return html
