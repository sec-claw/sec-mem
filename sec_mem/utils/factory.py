import importlib
from typing import Dict, Optional, Union

from sec_mem.configs.embeddings.base import BaseEmbedderConfig
from sec_mem.configs.llms.base import BaseLlmConfig
from sec_mem.configs.llms.ollama import OllamaConfig
from sec_mem.configs.llms.openai import OpenAIConfig
from sec_mem.embeddings.mock import MockEmbeddings


def load_class(class_type):
    module_path, class_name = class_type.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


class LlmFactory:
    """Factory for creating LLM instances."""

    provider_to_class = {
        "ollama": ("sec_mem.llms.ollama.OllamaLLM", OllamaConfig),
        "openai": ("sec_mem.llms.openai.OpenAILLM", OpenAIConfig),
    }

    @classmethod
    def create(cls, provider_name: str, config: Optional[Union[BaseLlmConfig, Dict]] = None, **kwargs):
        if provider_name not in cls.provider_to_class:
            raise ValueError(f"Unsupported LLM provider: {provider_name}")

        class_type, config_class = cls.provider_to_class[provider_name]
        llm_class = load_class(class_type)

        if config is None:
            config = config_class(**kwargs)
        elif isinstance(config, dict):
            config = config_class(**config, **kwargs)

        return llm_class(config)


class EmbedderFactory:
    """Factory for creating embedding instances."""

    provider_to_class = {
        "openai": "sec_mem.embeddings.openai.OpenAIEmbedding",
        "ollama": "sec_mem.embeddings.ollama.OllamaEmbedding",
    }

    @classmethod
    def create(cls, provider_name, config, vector_config: Optional[dict] = None):
        if provider_name == "upstash_vector" and vector_config and vector_config.enable_embeddings:
            return MockEmbeddings()
        
        class_type = cls.provider_to_class.get(provider_name)
        if class_type:
            embedder_instance = load_class(class_type)
            base_config = BaseEmbedderConfig(**config)
            return embedder_instance(base_config)
        else:
            raise ValueError(f"Unsupported Embedder provider: {provider_name}")


class VectorStoreFactory:
    """Factory for creating vector store instances."""

    provider_to_class = {
        "faiss": "sec_mem.vector_stores.faiss.FAISS",
        "faiss_advanced": "sec_mem.vector_stores.faiss_advanced.FAISSAdvanced",
    }

    @classmethod
    def create(cls, provider_name, config):
        class_type = cls.provider_to_class.get(provider_name)
        if class_type:
            if not isinstance(config, dict):
                config = config.model_dump()
            vector_store_instance = load_class(class_type)
            return vector_store_instance(**config)
        else:
            raise ValueError(f"Unsupported VectorStore provider: {provider_name}")

    @classmethod
    def reset(cls, instance):
        instance.reset()
        return instance


class GraphStoreFactory:
    """Factory for creating graph store instances (minimal)."""
    
    @classmethod
    def create(cls, provider_name, config):
        return None


class RerankerFactory:
    """Factory for creating reranker instances (minimal)."""
    
    @classmethod
    def create(cls, provider_name: str, config=None, **kwargs):
        raise ValueError(f"Reranker not supported in minimal build")
