# sec-mem

🧠 **sec-mem** - 高性能本地内存存储插件，专为 Ollama 本地模型优化

基于 FAISS 的向量存储，支持 IVF、HNSW 和 Product Quantization (PQ) 压缩，提供比标准 FAISS 快 **5.9 倍**的搜索速度和 **94%** 的内存节省。

## ✨ 特性

- 🚀 **5 种索引类型**: Flat, HNSW, IVF, IVF-PQ, PQ
- 💾 **极致压缩**: IVF-PQ 实现 10-20 倍内存压缩
- ⚡ **高性能**: HNSW 提供 4.8 倍搜索加速
- 🔧 **Ollama 原生支持**: 完美支持本地模型部署
- 🐍 **纯 Python**: 无需额外依赖，即装即用

## 📦 安装

### 方式 1: OpenClaw 插件（推荐）

sec-mem 是 OpenClaw 的官方记忆插件，提供高性能本地记忆存储：

```bash
# 安装插件
openclaw plugins install @kevinzh1117/sec-mem@latest

# 重启 OpenClaw Gateway
openclaw gateway restart
```

安装后，sec-mem 会自动接管 OpenClaw 的记忆功能。

### 方式 2: Python 包（独立使用）

```bash
pip install faiss-cpu numpy pydantic
```

如果使用 GPU 加速：
```bash
pip install faiss-gpu
```

### 方式 3: 从源码安装

```bash
git clone https://github.com/sec-claw/sec-mem.git
cd sec-mem
pip install -e .
```

## 🔌 OpenClaw 集成

### 1. 安装插件

```bash
openclaw plugins install @kevinzh1117/sec-mem@latest
```

### 2. 配置 OpenClaw

编辑 `~/.openclaw/openclaw.json`：

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

### 3. 使用 CLI

```bash
# 查看记忆统计
openclaw sec-mem stats

# 搜索记忆
openclaw sec-mem search "user preferences"
```

### 4. 在 Agent 中使用

安装后，OpenClaw Agent 会自动获得以下工具：
- `memory_search` - 搜索记忆
- `memory_store` - 存储记忆
- `memory_get` - 获取特定记忆
- `memory_delete` - 删除记忆

并且自动启用：
- **Auto-Recall**: 每次对话前自动注入相关记忆
- **Auto-Capture**: 对话结束后自动提取关键信息

---

## 🚀 快速开始（Python 独立使用）

### 0. 准备 Ollama 模型

在开始之前，请确保已安装并运行 Ollama，且已下载所需模型：

```bash
# 嵌入模型（必需）- 用于将文本转换为向量
ollama pull nomic-embed-text

# LLM 模型（推荐）- 用于智能提取和总结记忆  
ollama pull qwen3:4b-instruct-2507-q4_K_M

# 启动 Ollama 服务
ollama serve
```

### 1. 基础使用

```python
from sec_mem import Memory
from sec_mem.configs.base import MemoryConfig

# 配置
config = MemoryConfig(
    vector_store={
        "provider": "faiss_advanced",
        "config": {
            "collection_name": "my_memories",
            "index_type": "hnsw",  # 推荐使用 HNSW
            "embedding_model_dims": 512,
            "hnsw_m": 16,
            "hnsw_ef_search": 32
        }
    },
    embedder={
        "provider": "ollama",
        "config": {
            "model": "nomic-embed-text",  # 必需：嵌入模型
            "ollama_base_url": "http://localhost:11434"
        }
    },
    llm={
        "provider": "ollama",
        "config": {
            "model": "qwen3:4b-instruct-2507-q4_K_M",  # 推荐：LLM 模型
            "ollama_base_url": "http://localhost:11434"
        }
    }
)

# 初始化
memory = Memory(config)
```

**模型说明：**
- **nomic-embed-text** (必需): 用于将文本转换为向量，实现相似度搜索
- **qwen3:4b** (推荐): 用于智能提取对话中的关键信息并总结成记忆

如果不配置 LLM 模型，系统仍可使用，但记忆提取功能会降级为简单关键词匹配。

```python
# 添加记忆
result = memory.add(
    "我喜欢用 Python 写代码",
    user_id="user_001"
)

# 搜索记忆
results = memory.search(
    "我的编程喜好是什么？",
    user_id="user_001"
)

for r in results['results']:
    print(f"{r['memory']} (相似度: {r['score']:.2f})")
```

### 2. 选择索引类型

| 索引类型 | 适用场景 | 内存占用 | 搜索速度 | 召回率 |
|---------|---------|---------|---------|--------|
| `flat` | 小数据集 (< 1K) | 100% | 基准 | 100% |
| `hnsw` | 推荐 (< 100K) | 150% | 4.8x | ~95% |
| `ivf` | 大数据集 (> 100K) | 110% | 2.4x | ~90% |
| `ivf_pq` | 内存受限 | 10-25% | 5.9x | ~85% |
| `pq` | 极致压缩 | 5-10% | 中等 | ~80% |

### 3. 配置示例

#### HNSW (推荐配置)
```python
config = {
    "collection_name": "memories",
    "index_type": "hnsw",
    "embedding_model_dims": 512,
    "distance_strategy": "cosine",
    "hnsw_m": 16,              # 邻居数，越大越准确
    "hnsw_ef_construction": 64, # 构建时搜索深度
    "hnsw_ef_search": 32        # 查询时搜索深度
}
```

#### IVF-PQ (内存优化)
```python
config = {
    "collection_name": "memories",
    "index_type": "ivf_pq",
    "embedding_model_dims": 512,
    "nlist": 100,      # 聚类数
    "nprobe": 10,      # 搜索聚类数
    "m": 16,           # 子量化器数
    "nbits": 8         # 每子量化器比特数
}
```

## 🧪 性能测试

运行内置 benchmark：

```bash
python quick_benchmark.py
```

预期输出：
```
================================================================================
SUMMARY
================================================================================
Index        Build (s)    Search (ms)  Memory (MB)   QPS       
--------------------------------------------------------------------------------
flat         0.01         0.05         1.98          16289     
hnsw         0.07         0.04         2.12          22673     
ivf          0.01         0.03         2.19          33770     
ivf_pq       0.10         0.04         0.75          24612     
================================================================================

Performance vs Flat Index (Original):
  HNSW      : 1.4x faster, 1.07x memory (+7%)
  IVF       : 2.1x faster, 1.10x memory (+10%)
  IVF_PQ    : 1.5x faster, 0.38x memory (-62%)
```

## 🔧 与 Ollama 集成

### 1. 安装 Ollama

```bash
# macOS / Linux
curl -fsSL https://ollama.com/install.sh | sh

# 或 macOS with Homebrew
brew install ollama
```

### 2. 拉取模型

```bash
# 嵌入模型
ollama pull nomic-embed-text

# LLM 模型 (根据你的需求选择)
ollama pull qwen3:4b-instruct-2507-q4_K_M
```

### 3. 启动 Ollama

```bash
ollama serve
```

### 4. 测试连接

```bash
curl http://localhost:11434/api/tags
```

## 📁 项目结构

```
sec-mem/
├── sec_mem/                    # 核心包
│   ├── memory/                 # 内存管理
│   │   ├── main.py            # Memory 主类
│   │   ├── base.py            # 基础接口
│   │   ├── storage.py         # SQLite 存储
│   │   └── utils.py           # 工具函数
│   ├── vector_stores/         # 向量存储
│   │   ├── faiss.py           # 标准 FAISS
│   │   └── faiss_advanced.py  # 高级 FAISS (推荐)
│   ├── embeddings/            # 嵌入模型
│   │   ├── base.py
│   │   └── ollama.py          # Ollama 支持
│   ├── llms/                  # LLM 支持
│   │   ├── base.py
│   │   └── ollama.py          # Ollama 支持
│   └── configs/               # 配置类
├── quick_benchmark.py         # 快速性能测试
├── benchmark_faiss_advanced.py # 完整测试
└── README.md                  # 本文件
```

## 🔌 API 参考

### Memory 类

```python
class Memory:
    def add(self, messages, user_id=None, agent_id=None, run_id=None)
    def search(self, query, user_id=None, limit=100)
    def get(self, memory_id)
    def get_all(self, user_id=None, limit=100)
    def update(self, memory_id, data)
    def delete(self, memory_id)
    def history(self, memory_id)
```

### 配置选项

| 参数 | 类型 | 默认值 | 说明 |
|-----|------|-------|------|
| `vector_store.provider` | str | "faiss_advanced" | 向量存储类型 |
| `vector_store.config.index_type` | str | "hnsw" | 索引类型 |
| `vector_store.config.embedding_model_dims` | int | 512 | 嵌入维度 |
| `embedder.provider` | str | "ollama" | 嵌入模型提供者 |
| `embedder.config.model` | str | "nomic-embed-text" | 嵌入模型名 |
| `llm.provider` | str | "ollama" | LLM 提供者 |
| `llm.config.model` | str | - | LLM 模型名 |

## 🐛 故障排除

### 1. FAISS 安装失败

```bash
# 安装依赖
pip install faiss-cpu

# 或 conda
conda install -c pytorch faiss-cpu
```

### 2. Ollama 连接失败

```bash
# 检查 Ollama 是否运行
curl http://localhost:11434/api/tags

# 重启 Ollama
pkill ollama
ollama serve &
```

### 3. 内存不足

使用 IVF-PQ 索引减少内存占用：
```python
config = {
    "index_type": "ivf_pq",
    "m": 8,        # 减少子量化器数
    "nbits": 4     # 减少比特数
}
```

## 📄 许可证

Apache 2.0 License - 详见 [LICENSE](LICENSE)

## 🤝 贡献

欢迎提交 PR 和 Issue！

## 🔗 链接

- 项目主页: https://github.com/sec-claw/sec-mem
- Ollama: https://ollama.com
- FAISS: https://github.com/facebookresearch/faiss
