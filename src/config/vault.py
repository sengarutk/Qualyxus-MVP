from __future__ import annotations

import copy
import logging
import threading
from typing import Any

from src.config.config import Config

logger = logging.getLogger("QualyxusVault")

class ConfigVault:
    """
    Thread-Safe Configuration Proxy Layer.
    Enforces a strict, low-level object allocation override to guarantee
    a singular immutable configuration baseline across all edge threads.
    """
    _instance: ConfigVault | None = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs) -> ConfigVault:
        """
        Overrides low-level object allocation.
        Enforces a true double-checked locking singleton pattern even if the class
        is called directly via constructor syntax: ConfigVault().
        """
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    instance = super().__new__(cls)
                    # Flaw Fix: Route kwargs securely into the configuration loader to prevent silent parameter ignoring.
                    instance._underlying_config = Config.load(**kwargs)
                    cls._instance = instance
                elif kwargs:
                    # Positive Mod: Warn if a subsequent caller tries to inject new initialization parameters.
                    logger.warning("[ConfigVault] Ignoring kwargs on subsequent allocation. Singleton is already sealed.")
        elif kwargs:
            logger.warning("[ConfigVault] Ignoring kwargs on subsequent allocation. Singleton is already sealed.")
            
        return cls._instance

    def __init__(self, *args, **kwargs):
        """
        Flaw Fix: Absorb instantiation arguments so Python's automatic `object.__init__` 
        doesn't throw a TypeError when constructor arguments are used.
        """
        pass

    @classmethod
    def initialize(cls, **kwargs) -> ConfigVault:
        """Standard initialization interface mapping cleanly to upstream orchestrators."""
        return cls(**kwargs)

    def get(self, *args, default: Any = None) -> Any:
        """
        Lock-free high-velocity retrieval.
        Safe against lock-contention stalling because the underlying configuration is read-only.
        """
        # Intercept queries for VLM parameters to support fallback
        if len(args) >= 1 and args[0] in ("enabled", "model", "local_api", "timeout_seconds", "backend"):
            key = args[0]
            default_val = args[1] if len(args) > 1 else default
            
            # Check if it exists at root level first
            val = self._underlying_config.get(key, default=None)
            if val is None:
                # Fallback to vlm_reasoning sub-dict
                vlm_cfg = self._underlying_config.get("vlm_reasoning", default={})
                if isinstance(vlm_cfg, dict):
                    val = vlm_cfg.get(key, default_val)
                else:
                    val = default_val
            if val is None:
                val = default_val
        else:
            val = self._underlying_config.get(*args, default=default)

        if isinstance(val, (dict, list, set)):
            return copy.deepcopy(val)
        return val

    @property
    def raw(self) -> dict:
        """Lock-free access to the raw configuration map protected via deep copy isolation."""
        return copy.deepcopy(self._underlying_config.raw)
