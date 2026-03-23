import importlib.metadata

try:
    __version__ = importlib.metadata.version("sec-mem")
except importlib.metadata.PackageNotFoundError:
    __version__ = "1.0.0"

from sec_mem.memory.main import AsyncMemory, Memory  # noqa
