import asyncio

from prompt_toolkit import PromptSession
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.key_binding import KeyBindings, KeyPressEvent
from prompt_toolkit.keys import Keys
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.text import Text

from agent import Agent
from config import MODEL_NAME, OPENAI_BASE_URL

console = Console()

HELP_TEXT = """可用命令:
  /help   — 显示此帮助信息
  /clear  — 清除对话历史
  /exit   — 退出程序

支持直接粘贴多行内容。手动换行请按 Alt+Enter / Option+Enter。"""


def _build_key_bindings() -> KeyBindings:
    """
    按键绑定：
    - Enter:     提交输入（无论单行还是多行）
    - Alt+Enter: 手动插入换行
    粘贴多行时 prompt_toolkit 的 bracketed paste 会把所有内容
    （包括换行）一次性写入缓冲区，不会触发 Enter 按键事件，
    粘贴完成后用户按 Enter 提交即可。
    """
    kb = KeyBindings()

    @kb.add(Keys.Enter, eager=True)
    def handle_enter(event: KeyPressEvent) -> None:
        event.current_buffer.validate_and_handle()

    @kb.add(Keys.Escape, Keys.Enter, eager=True)  # Alt+Enter
    def handle_alt_enter(event: KeyPressEvent) -> None:
        event.current_buffer.insert_text("\n")

    return kb


def _create_session() -> PromptSession:
    return PromptSession(
        key_bindings=_build_key_bindings(),
        # multiline=False 让默认行为是单行，但我们的自定义绑定
        # 和 bracketed paste 仍然能正确处理多行粘贴内容
        multiline=False,
        enable_open_in_editor=False,
    )


def print_banner() -> None:
    console.print(
        Panel(
            Text.from_markup(
                f"[bold cyan]SearchBot[/] — 联网 AI Agent\n"
                f"[dim]模型: {MODEL_NAME} | 端点: {OPENAI_BASE_URL}[/]"
            ),
            border_style="cyan",
        )
    )
    console.print("[dim]输入 /help 查看可用命令 | Alt+Enter 换行 | Enter 提交[/]\n")


async def main() -> None:
    print_banner()
    agent = Agent()
    session = _create_session()

    while True:
        try:
            user_input = await session.prompt_async(HTML("<ansigreen><b>&gt; </b></ansigreen>"))
            user_input = user_input.strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]再见！[/]")
            break

        if not user_input:
            continue

        if user_input.lower() == "/exit":
            console.print("[dim]再见！[/]")
            break
        elif user_input.lower() == "/help":
            console.print(Markdown(HELP_TEXT))
            continue
        elif user_input.lower() == "/clear":
            agent.clear_history()
            console.print("[dim]对话历史已清除。[/]\n")
            continue

        console.print()
        full_text = ""
        try:
            async for token in agent.run(user_input):
                if token.startswith("\n🔧"):
                    if full_text:
                        console.print(Markdown(full_text))
                        full_text = ""
                    console.print(Text(token.strip(), style="bold yellow"))
                else:
                    full_text += token
        except Exception as e:
            console.print(f"[bold red]错误:[/] {e}")
            continue

        if full_text:
            console.print(Markdown(full_text))
        console.print()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
