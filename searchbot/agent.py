import asyncio
import json
import re
from typing import AsyncGenerator, Callable

from openai import AsyncOpenAI

from config import OPENAI_API_KEY, OPENAI_BASE_URL, MODEL_NAME, SYSTEM_PROMPT, logger
from tools import scrape_url, google_search, deep_explore

# 匹配纯 URL 输入（整条消息就是一个链接，前后可有空白）
_PURE_URL_RE = re.compile(r"^\s*(https?://\S+)\s*$", re.IGNORECASE)

TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "google_search",
            "description": "搜索 Google，返回相关网页的标题、URL 和摘要。适用于需要查找信息、了解最新动态的场景。",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "搜索关键词",
                    },
                    "num_results": {
                        "type": "integer",
                        "description": "返回结果数量，默认 5",
                        "default": 5,
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "scrape_url",
            "description": "爬取指定 URL 的网页内容，提取正文文本。适用于需要阅读特定网页内容的场景。",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "要爬取的网页 URL",
                    },
                },
                "required": ["url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "deep_explore",
            "description": "深度探索某个网站：爬取目标页面及其子页面，针对用户问题进行综合分析。适用于需要深入了解某个网站特定主题的场景。",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "要深度探索的起始 URL",
                    },
                    "question": {
                        "type": "string",
                        "description": "用户想要了解的具体问题",
                    },
                },
                "required": ["url", "question"],
            },
        },
    },
]

TOOL_FUNCTIONS = {
    "google_search": google_search,
    "scrape_url": scrape_url,
    "deep_explore": deep_explore,
}


class Agent:
    def __init__(self) -> None:
        self.client = AsyncOpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)
        self.messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]

    def clear_history(self) -> None:
        self.messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    async def _call_tool(self, name: str, arguments: str) -> str:
        func = TOOL_FUNCTIONS.get(name)
        if not func:
            logger.warning("未知工具调用: %s", name)
            return f"未知工具: {name}"
        try:
            args = json.loads(arguments)
            logger.info("执行工具 %s, 参数: %s", name, json.dumps(args, ensure_ascii=False))
            result = await func(**args)
            logger.debug("工具 %s 返回结果 (%d 字符): %s", name, len(result), result[:500])
            return result
        except Exception as e:
            logger.error("工具 %s 执行异常: %s", name, e, exc_info=True)
            return f"工具 {name} 执行出错: {e}"

    def _log_messages(self, tag: str) -> None:
        """记录当前完整 messages 链到日志。"""
        for i, msg in enumerate(self.messages):
            role = msg["role"]
            if role == "tool":
                content = msg["content"]
                logger.debug(
                    "[%s] messages[%d] role=%s tool_call_id=%s content(%d 字符)=%.200s",
                    tag, i, role, msg.get("tool_call_id", ""), len(content), content,
                )
            elif "tool_calls" in msg:
                calls = [
                    {"name": tc["function"]["name"], "arguments": tc["function"]["arguments"]}
                    for tc in msg["tool_calls"]
                ]
                logger.debug(
                    "[%s] messages[%d] role=%s content=%s tool_calls=%s",
                    tag, i, role, (msg.get("content") or "")[:200],
                    json.dumps(calls, ensure_ascii=False),
                )
            else:
                content = msg.get("content") or ""
                logger.debug(
                    "[%s] messages[%d] role=%s content(%d 字符)=%.500s",
                    tag, i, role, len(content), content,
                )

    async def run(self, user_input: str) -> AsyncGenerator[str, None]:
        """处理用户输入，流式返回最终回答。内部自动处理 tool calling 循环。"""
        self.messages.append({"role": "user", "content": user_input})
        logger.info("用户输入: %s", user_input)

        # ── 快速路径：纯 URL 输入，跳过 Agent 循环，直接爬取+总结 ──
        url_match = _PURE_URL_RE.match(user_input)
        if url_match:
            url = url_match.group(1)
            logger.info("检测到纯 URL 输入，直接爬取: %s", url)
            yield "\n🔧 调用工具: scrape_url\n"
            content = await scrape_url(url=url)
            logger.info("爬取完成, 内容长度=%d 字符", len(content))

            # 将爬取结果作为上下文，让 LLM 流式总结
            self.messages.append({
                "role": "assistant",
                "content": f"我已经爬取了该网页的内容，以下是网页内容：\n\n{content}",
            })
            self.messages.append({
                "role": "user",
                "content": "请根据上面的网页内容，给出详细的中文总结和要点分析。",
            })

            stream = await self.client.chat.completions.create(
                model=MODEL_NAME,
                messages=self.messages,
                stream=True,
            )
            full_response = ""
            async for chunk in stream:
                delta = chunk.choices[0].delta if chunk.choices else None
                if delta and delta.content:
                    full_response += delta.content
                    yield delta.content

            # 清理掉中间辅助消息，只保留最终回答
            self.messages.pop()  # 移除 "请总结" 的 user 消息
            self.messages.pop()  # 移除爬取内容的 assistant 消息
            self.messages.append({"role": "assistant", "content": full_response})
            logger.info("纯 URL 快速路径完成, 回答长度=%d 字符", len(full_response))
            return

        turn = 0
        while True:
            turn += 1
            logger.info("=== Agent 轮次 %d 开始 ===", turn)
            self._log_messages(f"turn-{turn}-request")

            # 先用非流式请求检查是否有 tool_calls
            response = await self.client.chat.completions.create(
                model=MODEL_NAME,
                messages=self.messages,
                tools=TOOLS_SCHEMA,
                tool_choice="auto",
            )

            choice = response.choices[0]
            message = choice.message
            logger.info(
                "LLM 响应: finish_reason=%s, has_tool_calls=%s, content_length=%d, usage=%s",
                choice.finish_reason,
                bool(message.tool_calls),
                len(message.content or ""),
                json.dumps(
                    {"prompt": response.usage.prompt_tokens, "completion": response.usage.completion_tokens, "total": response.usage.total_tokens}
                    if response.usage else {},
                ),
            )
            if message.content:
                logger.debug("LLM 文本内容: %.500s", message.content)

            # 如果有 tool_calls，执行工具并继续循环
            if message.tool_calls:
                tool_calls_data = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in message.tool_calls
                ]
                logger.info(
                    "LLM 请求工具调用: %s",
                    json.dumps(
                        [{"name": tc.function.name, "arguments": tc.function.arguments} for tc in message.tool_calls],
                        ensure_ascii=False,
                    ),
                )

                # 将 assistant 消息（含 tool_calls）加入历史
                self.messages.append(
                    {
                        "role": "assistant",
                        "content": message.content or "",
                        "tool_calls": tool_calls_data,
                    }
                )

                # 执行每个工具调用
                for tc in message.tool_calls:
                    yield f"\n🔧 调用工具: {tc.function.name}\n"
                    result = await self._call_tool(
                        tc.function.name, tc.function.arguments
                    )
                    self.messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "content": result,
                        }
                    )
                    logger.info(
                        "工具结果已加入消息链: tool_call_id=%s, result_length=%d",
                        tc.id, len(result),
                    )
                continue

            # 没有 tool_calls，使用流式输出最终回答
            logger.info("LLM 返回最终回答，切换到流式输出")
            stream = await self.client.chat.completions.create(
                model=MODEL_NAME,
                messages=self.messages,
                stream=True,
            )

            full_response = ""
            async for chunk in stream:
                delta = chunk.choices[0].delta if chunk.choices else None
                if delta and delta.content:
                    full_response += delta.content
                    yield delta.content

            self.messages.append({"role": "assistant", "content": full_response})
            logger.info("最终回答完成, 长度=%d 字符", len(full_response))
            logger.debug("最终回答内容: %.1000s", full_response)
            break

    async def chat(
        self,
        user_input: str,
        on_tool_call: Callable[[str, dict], None] | None = None,
    ) -> str:
        """
        一次调用，返回完整回答字符串。内部自动处理 tool calling 循环。

        参数:
            user_input:    用户输入文本
            on_tool_call:  可选回调，工具被调用时触发，签名 (tool_name, arguments) -> None

        返回:
            LLM 的最终回答文本

        用法:
            agent = Agent()
            answer = await agent.chat("搜索 Python 3.13 新特性")
            print(answer)
        """
        result_parts: list[str] = []
        async for token in self.run(user_input):
            if token.startswith("\n🔧"):
                # 工具调用提示，触发回调
                if on_tool_call:
                    # 从提示文本中提取工具名
                    tool_name = token.strip().removeprefix("🔧 调用工具:").strip()
                    on_tool_call(tool_name, {})
            else:
                result_parts.append(token)
        return "".join(result_parts)


def chat_sync(
    message: str,
    *,
    on_tool_call: Callable[[str, dict], None] | None = None,
    keep_session: bool = False,
    _agent: list | None = None,
) -> str:
    """
    同步接口 — 最简单的调用方式，一行代码即可使用。

    参数:
        message:       用户输入
        on_tool_call:  可选回调，工具被调用时触发
        keep_session:  True 保持多轮对话上下文，False 每次独立对话

    用法:
        from agent import chat_sync
        print(chat_sync("搜索 Python 3.13 新特性"))
    """
    if _agent is None:
        _agent = []
    if not _agent or not keep_session:
        _agent.clear()
        _agent.append(Agent())
    return asyncio.run(_agent[0].chat(message, on_tool_call=on_tool_call))
