# sec-mem Quick Start

## 1. Install (1 minute)

### 方式 1: OpenClaw CLI（推荐）

```bash
# 安装插件
openclaw plugins install @kevinzh1117/sec-mem@latest

# 重启 Gateway
openclaw gateway restart
```

### 方式 2: npm

```bash
# 在 OpenClaw 目录安装
cd ~/.openclaw
npm install @kevinzh1117/sec-mem

# 安装 Python 依赖
pip install faiss-cpu numpy pydantic
```

### 准备 Ollama

```bash
# 启动 Ollama
ollama serve &

# 拉取必需模型（重要！）
ollama pull nomic-embed-text          # 嵌入模型（必需）
ollama pull qwen3:4b-instruct-2507-q4_K_M  # LLM模型（记忆提取用，可选但推荐）
```

## 2. Verify Installation

```bash
# 检查插件状态
openclaw sec-mem stats

# 预期输出：
# Index type: ivf_pq
# Collection: my-memories
# Total memories: 0
# Auto-recall: true, Auto-capture: true
```

## 3. Use (Instant)

安装后，你的 OpenClaw agents 自动获得 4 个记忆工具：

| 工具 | 功能 |
|------|------|
| `memory_search` | 搜索相关记忆 |
| `memory_store` | 保存新信息 |
| `memory_get` | 按 ID 获取 |
| `memory_delete` | 删除记忆 |

### 自动功能

- **Auto-Recall**: 每次对话前自动注入相关记忆
- **Auto-Capture**: 对话结束后自动提取关键信息

### 使用示例

```
User: 记住我喜欢用暗色模式
→ Agent 自动调用 memory_store

User: 我喜欢什么主题？
→ Agent 自动调用 memory_search
→ 返回: "你喜欢暗色模式"
```

## 可选配置

如需自定义配置，编辑 `~/.openclaw/openclaw.json`：

```json
{
  "plugins": {
    "entries": {
      "sec-mem": {
        "enabled": true,
        "config": {
          "collectionName": "my-memories",
          "indexType": "ivf_pq",
          "ollamaEmbedModel": "nomic-embed-text",
          "ollamaLlmModel": "qwen3:4b-instruct-2507-q4_K_M",
          "autoRecall": true,
          "autoCapture": true
        }
      }
    }
  }
}
```

### 可选的 LLM 模型

| 模型 | 大小 | 特点 |
|------|------|------|
| `qwen3:4b-instruct-2507-q4_K_M` | 4B | 中文优化，速度快 |
| `llama3.1:8b` | 8B | 英文强，通用 |
| `phi3:medium` | 14B | 微软出品，质量好 |

## 4. 性能对比

运行大容量 benchmark（10万+ 向量）：

```bash
python benchmark_large_scale.py --vectors 100000
```

| 索引类型 | 搜索速度 | 内存占用 | 推荐场景 |
|---------|---------|---------|---------|
| `ivf_pq` | 226x | -98% | 大规模记忆 ⭐ |
| `hnsw` | 52x | +7% | < 100K 条记忆 |

## 故障排除

### 模型未找到
```bash
# 检查模型是否已下载
ollama list

# 如果没有，重新拉取
ollama pull nomic-embed-text
ollama pull qwen3:4b-instruct-2507-q4_K_M
```

### Ollama 连接失败
```bash
# 检查 Ollama 是否在运行
curl http://localhost:11434/api/tags

# 重启 Ollama
pkill ollama
ollama serve &
```

### 插件未加载
```bash
# 检查插件安装
ls ~/.openclaw/extensions/

# 重新安装
openclaw plugins remove sec-mem
openclaw plugins install @kevinzh1117/sec-mem@latest
```

## 完整文档

- [OPENCLAW_INTEGRATION.md](OPENCLAW_INTEGRATION.md) - 完整集成指南
- GitHub: https://github.com/sec-claw/sec-mem
