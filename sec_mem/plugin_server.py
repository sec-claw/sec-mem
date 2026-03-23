#!/usr/bin/env python3
"""
sec-mem Plugin Server for OpenClaw

Handles IPC between Node.js plugin and Python backend.
"""

import sys
import json
import os
import warnings

# Suppress FAISS warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sec_mem.memory.main import Memory
from sec_mem.configs.base import MemoryConfig


class PluginServer:
    def __init__(self, config):
        self.config = config
        self.memory = self._init_memory()
        
    def _init_memory(self):
        """Initialize Memory instance with config."""
        memory_config = MemoryConfig(
            vector_store={
                "provider": "faiss_advanced",
                "config": {
                    "collection_name": self.config.get("collectionName", "sec-mem"),
                    "index_type": self.config.get("indexType", "hnsw"),
                    "embedding_model_dims": self.config.get("embeddingDims", 512),
                    "distance_strategy": self.config.get("distanceStrategy", "cosine"),
                    "path": self.config.get("storagePath", "./sec-mem-data"),
                    # HNSW params
                    "hnsw_m": self.config.get("hnswM", 16),
                    "hnsw_ef_construction": self.config.get("hnswEfConstruction", 64),
                    "hnsw_ef_search": self.config.get("hnswEfSearch", 32),
                    # IVF params
                    "nlist": self.config.get("nlist", 100),
                    "nprobe": self.config.get("nprobe", 10),
                    # PQ params
                    "m": self.config.get("m", 16),
                    "nbits": self.config.get("nbits", 8),
                }
            },
            embedder={
                "provider": "ollama",
                "config": {
                    "model": self.config.get("ollamaEmbedModel", "nomic-embed-text"),
                    "ollama_base_url": self.config.get("ollamaUrl", "http://localhost:11434"),
                }
            },
            llm={
                "provider": "ollama",
                "config": {
                    "model": self.config.get("ollamaLlmModel", "qwen3:4b-instruct-2507-q4_K_M"),
                    "ollama_base_url": self.config.get("ollamaUrl", "http://localhost:11434"),
                }
            } if self.config.get("ollamaLlmModel") else None
        )
        
        return Memory(memory_config)
    
    def handle_command(self, command, params):
        """Handle incoming commands."""
        try:
            if command == "add":
                result = self.memory.add(
                    params["message"],
                    user_id=params.get("userId", "default"),
                    metadata=params.get("metadata")
                )
                # Normalize to mem0 format: { results: [...] }
                if isinstance(result, dict) and "id" in result:
                    normalized = {
                        "results": [{
                            "id": result.get("id", ""),
                            "memory": params.get("message", ""),
                            "event": result.get("event", "ADD")
                        }]
                    }
                elif isinstance(result, list):
                    normalized = {
                        "results": [
                            {
                                "id": r.get("id", r.get("memory_id", "")),
                                "memory": r.get("memory", r.get("text", "")),
                                "event": r.get("event", "ADD")
                            }
                            for r in result
                        ]
                    }
                else:
                    normalized = {"results": []}
                return {"success": True, "result": normalized}
            
            elif command == "search":
                result = self.memory.search(
                    params["query"],
                    user_id=params.get("userId", "default"),
                    limit=params.get("limit", 5)
                )
                # Normalize to { results: [...] } format
                if isinstance(result, dict) and "results" in result:
                    normalized = result
                elif isinstance(result, list):
                    normalized = {"results": [
                        {
                            "id": r.get("id", r.get("memory_id", "")),
                            "memory": r.get("memory", r.get("text", "")),
                            "score": r.get("score"),
                            "created_at": r.get("created_at", r.get("createdAt"))
                        }
                        for r in result
                    ]}
                else:
                    normalized = {"results": []}
                return {"success": True, "result": normalized}
            
            elif command == "get":
                result = self.memory.get(params["memoryId"])
                if result and isinstance(result, dict):
                    normalized = {
                        "id": result.get("id", result.get("memory_id", params["memoryId"])),
                        "memory": result.get("memory", result.get("text", "")),
                        "created_at": result.get("created_at", result.get("createdAt")),
                        "updated_at": result.get("updated_at", result.get("updatedAt"))
                    }
                else:
                    normalized = result
                return {"success": True, "result": normalized}
            
            elif command == "get_all":
                result = self.memory.get_all(
                    user_id=params.get("userId", "default"),
                    limit=params.get("limit", 100)
                )
                return {"success": True, "result": result}
            
            elif command == "update":
                result = self.memory.update(
                    params["memoryId"],
                    params["data"]
                )
                return {"success": True, "result": result}
            
            elif command == "delete":
                self.memory.delete(params["memoryId"])
                return {"success": True, "result": None}
            
            elif command == "stats":
                info = self.memory.vector_store.col_info()
                return {"success": True, "result": {
                    "indexType": info.get("index_type"),
                    "numVectors": info.get("count", 0),
                    "memoryUsage": info.get("memory_estimate_mb", 0)
                }}
            
            else:
                return {"success": False, "error": f"Unknown command: {command}"}
                
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def run(self):
        """Main server loop."""
        # Signal ready
        print("PLUGIN_READY", flush=True)
        
        while True:
            try:
                line = sys.stdin.readline()
                if not line:
                    break
                
                message = json.loads(line.strip())
                msg_id = message.get("id")
                command = message.get("command")
                params = message.get("params", {})
                
                response = self.handle_command(command, params)
                response["id"] = msg_id
                
                print(json.dumps(response), flush=True)
                
            except json.JSONDecodeError as e:
                print(json.dumps({"success": False, "error": f"Invalid JSON: {e}"}), flush=True)
            except Exception as e:
                print(json.dumps({"success": False, "error": str(e)}), flush=True)


def main():
    """Entry point."""
    if len(sys.argv) < 2:
        print("Usage: plugin_server.py <config_json>", file=sys.stderr)
        sys.exit(1)
    
    config = json.loads(sys.argv[1])
    server = PluginServer(config)
    server.run()


if __name__ == "__main__":
    main()
