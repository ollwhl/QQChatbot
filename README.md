# QQ 聊天机器人 - 数字分身

基于多阶段 LLM 流水线的 QQ 聊天机器人，可作为你的数字分身自动参与私聊和群聊对话。通过 NapCat QQ Bot Framework 接入 QQ。

## 架构

```
NapCat Webhook ──> Flask (server.py:5001) ──> SQLite 存储
                                    │
                          AI_agent.msg_manger()
                          (消息分析模型：判断是否需要回复)
                                    │ 需要回复
                          AI_agent.generate_chat_response()
                          (对话模型：生成符合人设的回复)
                                    │
                          qq_msg.send_group/private_message()
                          (通过 NapCat 发送)
```

**消息收集服务器** (`msg_server.py:5002`) 独立运行，负责将目标群/用户的消息持久化到数据库。

## 文件结构

| 文件 | 说明 |
|------|------|
| `server.py` | 聊天机器人主服务 (Flask, 端口 5001) |
| `msg_server.py` | 消息收集服务 (Flask, 端口 5002) |
| `AI_agent.py` | 两阶段 AI 决策流水线 |
| `call_llm.py` | LLM API 统一封装 (支持 OpenAI / Gemini / Anthropic) |
| `qq_msg.py` | NapCat QQ API 客户端 |
| `database.py` | SQLite 数据库管理 |
| `config.py` | 配置加载模块 |
| `config.json` | 全局配置文件 |
| `logger.py` | 日志管理 (按天轮转) |
| `*_prompt.txt` | AI 人设和决策规则的 Prompt 文件 |

## 快速开始

### 前置要求

- Python 3.10+
- NapCat QQ 客户端运行中

### 安装

```bash
pip install -r requirements.txt
```

### 配置

编辑 `config.json`：

```jsonc
{
    "MODEL": {
        "MESSAGE_ANALYZE_MODEL": {
            "API_TYPE": "openai",       // openai / gemini / anthropic
            "MODEL": "gpt-4o-mini",
            "BASE_URL": "https://...",
            "API_KEY": "",              // 直接填写，或留空使用环境变量
            "API_KEY_ENV": "GPT4OMINI_API_KEY"  // API_KEY 为空时读取此环境变量
        },
        // CHAT_MODEL, IMAGE_MODEL, SEARCH_MODEL 同理
    },
    "master_user_id": 2727873726,       // 机器人代表的 QQ 号
    "chatbot_server": {
        "NAPCAT_HOST": "http://127.0.0.1:3000",
        "NAPCAT_TOKEN": "your_token",
        "trun_on_cmd":"XX说话",         // 启动口令
        "turn_off_cmd":"XX闭嘴",        // 关闭口令
        "target_groups": [123456789],   // 目标群号
        "target_users": [123456789],    // 目标私聊用户
        "debug": true                   // true 时不实际发送消息
    },
    "message_server": {
        "target_groups": [],
        "target_users": [],
        "debug": true
    }
}
```

API 密钥可通过环境变量设置：

```bash
export GPT4OMINI_API_KEY="sk-..."
export DEEPSEEK_API_KEY="sk-..."
export API_GPT_KEY="sk-..."
```

### 运行

```bash
# 启动聊天机器人服务 (端口 5001)
python server.py

# 启动消息收集服务 (端口 5002)
python msg_server.py

```

Linux可以用 screen -dmS xxx 命令创建任务运行

在 NapCat 中将 HTTP POST Webhook 地址指向对应服务的端口。

## 数据库

SQLite 数据库 `chat_history.db`，`messages` 表：

- `msg_id` (主键)、`timestamp`、`group_id`、`user_id`
- `sender_name`、`sender_card`、`content`
- `is_ai`、`is_group`

## 日志

- `group_chat_ai.log` - 群聊 AI 交互日志
- `private_chat_ai.log` - 私聊 AI 交互日志
- 按天轮转，保留 7 天
