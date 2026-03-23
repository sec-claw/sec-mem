# OpenClaw Integration Guide

Complete guide for installing and using sec-mem plugin with OpenClaw.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Installation](#installation)
3. [Configuration](#configuration)
4. [Usage](#usage)
5. [Advanced Configuration](#advanced-configuration)
6. [Troubleshooting](#troubleshooting)

---

## Prerequisites

### System Requirements

- **Node.js**: >= 18.0.0
- **Python**: >= 3.9
- **Ollama**: Installed and running

### Install Ollama

```bash
# macOS
brew install ollama

# Or download from https://ollama.com
```

### Pull Required Models

```bash
# Start Ollama service
ollama serve &

# Pull embedding model
ollama pull nomic-embed-text

# Pull LLM model (choose one)
ollama pull qwen3:4b-instruct-2507-q4_K_M
# or
ollama pull llama3.1:8b
```

---

## Installation

### Step 1: Install sec-mem Plugin

**推荐方式 - 使用 OpenClaw CLI:**

```bash
# 安装插件
openclaw plugins install @kevinzh1117/sec-mem@latest

# 重启 Gateway 以加载插件
openclaw gateway restart
```

**备选方式 - 使用 npm:**

```bash
# 全局安装
npm install -g @kevinzh1117/sec-mem

# 或在 OpenClaw 目录安装
cd ~/.openclaw
npm install @kevinzh1117/sec-mem
```

### Step 2: Install Ollama Models

sec-mem 需要两个 Ollama 模型：

```bash
# 1. 嵌入模型（必需）- 用于向量编码
ollama pull nomic-embed-text

# 2. LLM 模型（可选但推荐）- 用于智能提取记忆
ollama pull qwen3:4b-instruct-2507-q4_K_M
# 或其他模型: llama3.1:8b, phi3:medium, etc.
```

### Step 3: Verify Python Dependencies

```bash
# Ensure Python dependencies are installed
pip install faiss-cpu numpy pydantic

# For GPU acceleration (optional)
pip install faiss-gpu
```

### Step 3: Test Installation

```bash
# Run the built-in benchmark
npx sec-mem-benchmark
```

Expected output:
```
==================================================
sec-mem Quick Benchmark
==================================================
Index        Build (s)    Search (ms)  Memory (MB)  QPS
--------------------------------------------------------------------------------
flat         0.01         0.05         1.98         16289
hnsw         0.07         0.04         2.12         22673     ✓ Recommended
ivf          0.01         0.03         2.19         33770
ivf_pq       0.10         0.04         0.75         24612     ✓ Compressed
```

---

## Configuration

### Basic Configuration

插件安装后会自动配置。如需自定义，编辑 `~/.openclaw/openclaw.json`：

```json
{
  "plugins": {
    "installs": {
      "sec-mem": {
        "source": "npm",
        "spec": "@kevinzh1117/sec-mem@latest"
      }
    },
    "slots": {
      "memory": "sec-mem"
    },
    "entries": {
      "sec-mem": {
        "enabled": true,
        "config": {
          "collectionName": "openclaw-memories",
          "indexType": "hnsw",
          "embeddingDims": 512,
          "storagePath": "./data/sec-mem",
          "ollamaUrl": "http://localhost:11434",
          "ollamaEmbedModel": "nomic-embed-text",
          "ollamaLlmModel": "qwen3:4b-instruct-2507-q4_K_M",
          "autoRecall": true,
          "autoCapture": true,
          "topK": 5,
          "searchThreshold": 0.5
        }
      }
    }
  }
}
```

**必需的配置：**
- `ollamaEmbedModel`: 嵌入模型名称（必需，用于向量编码）
- `ollamaLlmModel`: LLM 模型名称（可选但推荐，用于智能提取记忆）

**如果你只配置了 `ollamaEmbedModel`：**
- ✅ 记忆存储和搜索功能正常
- ⚠️ 记忆提取功能会降级（使用简单关键词提取）

### Configuration Options

| Option | 必需 | 类型 | 默认值 | 说明 |
|--------|------|------|--------|------|
| `ollamaEmbedModel` | ✅ | string | `"nomic-embed-text"` | **嵌入模型**（必需，用于向量编码） |
| `ollamaLlmModel` | ❌ | string | - | **LLM 模型**（推荐，用于智能记忆提取） |
| `collectionName` | ❌ | string | `"sec-mem"` | 集合名称 |
| `indexType` | ❌ | string | `"hnsw"` | 索引类型 |
| `embeddingDims` | ❌ | number | `512` | 嵌入维度（必须匹配模型） |
| `distanceStrategy` | ❌ | string | `"cosine"` | 距离度量 |
| `storagePath` | ❌ | string | `"./sec-mem-data"` | 存储路径 |
| `ollamaUrl` | ❌ | string | `"http://localhost:11434"` | Ollama 服务地址 |
| `autoRecall` | ❌ | boolean | `true` | 自动回忆 |
| `autoCapture` | ❌ | boolean | `true` | 自动捕获 |
| `topK` | ❌ | number | `5` | 返回记忆数量 |
| `searchThreshold` | ❌ | number | `0.5` | 相似度阈值 |

### Index Type Selection

Choose the right index type for your use case:

```json
// Fastest for < 100K vectors (Recommended)
{
  "indexType": "hnsw",
  "hnswM": 16,
  "hnswEfSearch": 32
}

// Balanced for > 100K vectors
{
  "indexType": "ivf",
  "nlist": 100,
  "nprobe": 10
}

// Maximum compression for memory-constrained
{
  "indexType": "ivf_pq",
  "nlist": 100,
  "nprobe": 10,
  "m": 16,
  "nbits": 8
}
```

---

## Usage

### Available Tools

Once installed, your OpenClaw agents can use these tools:

#### 1. memory_search

Search through stored memories.

```typescript
// Example tool call
{
  name: "memory_search",
  params: {
    query: "What are the user's preferences?",
    limit: 5,
    userId: "user_001"  // optional
  }
}
```

**Response:**
```
Found 3 memories:

1. User prefers dark mode for all applications (score: 95%)
2. User likes Python over JavaScript (score: 87%)
3. User works remotely from Tokyo (score: 82%)
```

#### 2. memory_store

Store new information.

```typescript
{
  name: "memory_store",
  params: {
    text: "User just got a promotion to Senior Engineer",
    userId: "user_001",
    metadata: { category: "career" }
  }
}
```

**Response:**
```
Memory stored successfully (ID: mem_abc123, Event: ADD)
```

#### 3. memory_get

Retrieve a specific memory by ID.

```typescript
{
  name: "memory_get",
  params: {
    memoryId: "mem_abc123"
  }
}
```

#### 4. memory_delete

Delete a memory.

```typescript
{
  name: "memory_delete",
  params: {
    memoryId: "mem_abc123"
  }
}
```

### Auto-Recall Feature

With `autoRecall: true`, sec-mem automatically searches for relevant memories before each agent turn:

```
User: "What should I work on today?"

[sec-mem auto-recall]
→ Found relevant memories about user's projects
→ Injected into context

Agent: "Based on your previous notes, you mentioned wanting to finish the 
        authentication module and review the API documentation."
```

---

## Advanced Configuration

### Multi-User Setup

```json
{
  "plugins": {
    "entries": {
      "sec-mem": {
        "enabled": true,
        "package": "@kevinzh1117/sec-mem",
        "config": {
          "collectionName": "multi-user-memories",
          "indexType": "hnsw",
          "storagePath": "./data/shared-sec-mem"
        }
      }
    }
  },
  "users": {
    "user_001": { "name": "Alice" },
    "user_002": { "name": "Bob" }
  }
}
```

### Performance Tuning

For production deployments:

```json
{
  "config": {
    "indexType": "ivf_pq",
    "nlist": 256,
    "nprobe": 20,
    "m": 32,
    "nbits": 8,
    "storagePath": "/var/lib/sec-mem"
  }
}
```

### Custom Ollama Configuration

If Ollama runs on a different host:

```json
{
  "config": {
    "ollamaUrl": "http://192.168.1.100:11434",
    "ollamaEmbedModel": "mxbai-embed-large",
    "embeddingDims": 768
  }
}
```

---

## Troubleshooting

### Plugin Not Loading

```bash
# Check if plugin is installed
npm list @kevinzh1117/sec-mem

# Reinstall if needed
npm install @kevinzh1117/sec-mem --force
```

### Ollama Connection Failed

```bash
# Test Ollama connection
curl http://localhost:11434/api/tags

# If failed, restart Ollama
pkill ollama
ollama serve &
```

### Python Module Not Found

```bash
# Install Python dependencies manually
pip install -e /path/to/node_modules/@kevinzh1117/sec-mem

# Or set PYTHONPATH
export PYTHONPATH="/path/to/node_modules/@kevinzh1117/sec-mem:$PYTHONPATH"
```

### Index Corruption

If the FAISS index becomes corrupted:

```bash
# Delete the storage directory and restart
rm -rf ./data/sec-mem

# The index will be rebuilt automatically
```

### Performance Issues

1. **Slow search**: Switch to `hnsw` index type
2. **High memory usage**: Use `ivf_pq` with lower `m` value
3. **Slow startup**: Reduce `nlist` for IVF indexes

---

## CLI Commands

If OpenClaw supports CLI extensions:

```bash
# Search memories
openclaw sec-mem search "user preferences"

# Show stats
openclaw sec-mem stats

# Rebuild index
openclaw sec-mem rebuild

# Export memories
openclaw sec-mem export --format json > memories.json
```

---

## Migration from Other Memory Plugins

### From mem0-openclaw

```bash
# Uninstall old plugin
npm uninstall @mem0/openclaw-mem0

# Install sec-mem
npm install @kevinzh1117/sec-mem

# Update config: change "@mem0/openclaw-mem0" to "@kevinzh1117/sec-mem"
```

Data migration:

```bash
# Export from old system
openclaw mem0 export > old_memories.json

# Import to sec-mem (script needed)
node scripts/migrate-mem0-to-sec-mem.js
```

---

## Support

- GitHub Issues: https://github.com/sec-claw/sec-mem/issues
- Documentation: https://github.com/sec-claw/sec-mem#readme
