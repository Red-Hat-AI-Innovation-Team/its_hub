"""
A Python library for inference-time scaling LLMs
"""

try:
    from importlib.metadata import version
    __version__ = version("its_hub")
except Exception:
    # Fallback version when package metadata is not available (e.g., during testing)
    __version__ = "0.0.0+dev" 