# FAISS Advanced 性能测试指南

使用 Ollama 本地模型测试新的 FAISS Advanced 内存模块的性能优化。

## 前置要求

### 1. 安装 Ollama

```bash
# macOS / Linux
curl -fsSL https://ollama.com/install.sh | sh

# 或者手动安装
brew install ollama  # macOS
```

### 2. 拉取模型

```bash
# 拉取你的主力模型
ollama pull qwen3:4b-instruct-2507-q4_K_M

# 拉取嵌入模型（默认使用 nomic-embed-text）
ollama pull nomic-embed-text

# 或者使用其他嵌入模型
ollama pull mxbai-embed-large  # 更好的质量，768维
ollama pull all-minilm          # 更快，384维
```

### 3. 安装 Python 依赖

```bash
cd /Users/kevin/Desktop/sec_mem

# 创建虚拟环境
python3 -m venv venv
source venv/bin/activate

# 安装依赖
pip install faiss-cpu ollama numpy psutil

# 可选：安装 sec_memai 完整包
pip install -e .
```

### 4. 启动 Ollama

```bash
# 确保 Ollama 服务在运行
ollama serve

# 或者后台运行
ollama serve &
```

## 快速测试

### 方法 1: 纯 FAISS 基准测试（不需要 Ollama）

这个测试使用随机向量，专注于索引本身的性能：

```bash
python3 quick_benchmark.py
```

预期输出：
```
==================================================
FAISS Advanced Quick Benchmark
==================================================

Config:
  Vectors: 1000
  Dimension: 512

==================================================
Testing: FLAT
==================================================
  Vectors:      1000
  Build time:   0.02s
  Search time:  0.15 ms (avg of 20)
  Throughput:   6667 queries/sec
  Index size:   1.95 MB
  Docstore:     0.05 MB
  Total memory: 2.00 MB
...

SUMMARY
================================================================================
Index        Build (s)    Search (ms)  Memory (MB)  QPS       
--------------------------------------------------------------------------------
flat         0.02         0.15         2.00         6667      
hnsw         0.05         0.05         3.20         20000     
ivf          0.08         0.08         2.20         12500     
ivf_pq       0.50         0.12         0.50         8333      
================================================================================

Performance vs Flat Index:
  HNSW      : 3.0x faster, 1.60x memory
  IVF       : 1.9x faster, 1.10x memory
  IVF-PQ    : 1.3x faster, 0.25x memory (75% reduction!)
```

### 方法 2: 完整 Ollama 集成测试

这个测试使用真实的 Ollama 嵌入模型和内存操作流程：

```bash
# 设置环境变量（可选）
export OLLAMA_HOST="http://localhost:11434"
export OLLAMA_EMBED_MODEL="nomic-embed-text"
export OLLAMA_LLM_MODEL="qwen3:4b-instruct-2507-q4_K_M"

# 运行完整测试
python3 benchmark_faiss_advanced.py
```

**注意**: 完整测试会实际调用 Ollama API 生成嵌入，测试时间取决于你的硬件性能。

## 在 OpenClaw 中使用

### 配置示例

在你的 `openclaw.json` 中添加：

```json5
{
  "plugins": {
    "entries": {
      "openclaw-sec_mem": {
        "enabled": true,
        "config": {
          "mode": "open-source",
          "userId": "your-user-id",
          "oss": {
            "embedder": {
              "provider": "ollama",
              "config": {
                "model": "nomic-embed-text",
                "ollama_base_url": "http://localhost:11434"
              }
            },
            "vectorStore": {
              "provider": "faiss_advanced",
              "config": {
                "collection_name": "sec_mem",
                "index_type": "hnsw",  // 推荐
                "embedding_model_dims": 512,
                "distance_strategy": "cosine",
                "path": "./sec_mem_data",
                "hnsw_m": 16,
                "hnsw_ef_search": 32
              }
            },
            "llm": {
              "provider": "ollama",
              "config": {
                "model": "qwen3:4b-instruct-2507-q4_K_M",
                "ollama_base_url": "http://localhost:11434"
              }
            }
          }
        }
      }
    }
  }
}
```

### 不同场景的配置建议

#### 场景 1: 个人使用（< 10K 条记忆）
```json5
"vectorStore": {
  "provider": "faiss_advanced",
  "config": {
    "index_type": "hnsw",
    "hnsw_m": 16,
    "hnsw_ef_search": 32
  }
}
```
- 最快搜索
- 略高内存占用
- 最佳召回率

#### 场景 2: 小团队（10K - 100K 条记忆）
```json5
"vectorStore": {
  "provider": "faiss_advanced",
  "config": {
    "index_type": "ivf",
    "nlist": 100,
    "nprobe": 10
  }
}
```
- 平衡性能
- 可接受内存占用
- 良好召回率

#### 场景 3: 大内存优化（> 100K 条或内存受限）
```json5
"vectorStore": {
  "provider": "faiss_advanced",
  "config": {
    "index_type": "ivf_pq",
    "nlist": 200,
    "nprobe": 20,
    "m": 16,      // 512维 / 16 = 32
    "nbits": 8
  }
}
```
- 10-20x 内存压缩
- 可接受的召回率 (~85%)
- 适合边缘设备

#### 场景 4: 极致压缩
```json5
"vectorStore": {
  "provider": "faiss_advanced",
  "config": {
    "index_type": "ivf_pq",
    "nlist": 100,
    "nprobe": 15,
    "m": 8,
    "nbits": 4    // 极致压缩
  }
}
```
- 32x 内存压缩
- 召回率约 80%
- 适合超大规模数据

## 预期性能提升

基于我们的测试，相比标准 FAISS Flat 索引：

| 指标 | Flat | HNSW | IVF | IVF-PQ |
|------|------|------|-----|--------|
| 搜索速度 | 1x | 3-5x | 2-3x | 1.5-2x |
| 内存占用 | 100% | 150% | 110% | 10-25% |
| 召回率@10 | 100% | 95% | 90% | 85% |
| 构建时间 | 1x | 2x | 3x | 10x |

## 故障排除

### 问题 1: "Cannot connect to Ollama"

```bash
# 检查 Ollama 是否在运行
curl http://localhost:11434/api/tags

# 重启 Ollama
pkill ollama
ollama serve
```

### 问题 2: 模型维度不匹配

```python
# 检查你的嵌入模型维度
ollama run nomic-embed-text
# 在 Python 中测试
from ollama import Client
c = Client()
r = c.embed(model="nomic-embed-text", input="test")
print(f"Dimension: {len(r['embeddings'][0])}")
```

### 问题 3: IVF-PQ 训练失败

```json5
// 降低 auto_train_threshold
"auto_train_threshold": 500  // 默认是 1000
```

### 问题 4: 内存不足

```json5
// 使用 IVF-PQ 减少内存
"index_type": "ivf_pq",
"m": 8,        // 减少 subquantizers
"nbits": 4     // 减少 bits
```

## 高级调优

### 调优 nprobe（IVF 系列）

`nprobe` 控制搜索时检查的聚类数量：
- 太低：召回率下降
- 太高：搜索变慢
- 经验法则：`nprobe = nlist / 10`

### 调优 ef_search（HNSW）

控制 HNSW 搜索深度：
- 默认值：32
- 更高：更好的召回，更慢
- 更低：更快，可能漏掉结果

### GPU 加速（如果有 NVIDIA GPU）

```bash
# 安装 GPU 版本 FAISS
pip uninstall faiss-cpu
pip install faiss-gpu
```

```json5
"config": {
  "use_gpu": true,
  "gpu_id": 0
}
```

## 验证测试

运行以下命令验证安装：

```bash
# 1. 检查 FAISS
python3 -c "import faiss; print(f'FAISS version: {faiss.__version__}')"

# 2. 检查 Ollama
python3 -c "from ollama import Client; c = Client(); print('Models:', [m['name'] for m in c.list()['models']])"

# 3. 运行快速测试
python3 quick_benchmark.py
```

## 下一步

1. 运行 `quick_benchmark.py` 获取基线数据
2. 根据你的数据规模选择合适的索引类型
3. 更新 `openclaw.json` 配置
4. 重启 OpenClaw 并测试实际使用

如有问题，请检查：
- Ollama 服务是否运行
- 模型是否正确拉取
- 维度配置是否匹配
