import logging
import threading
from typing import Any, Optional

logger = logging.getLogger("QualyxusFactory")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

class SafeServiceStub:
    """
    Defensive Pipeline Stubbing shield. 
    Neutralizes downstream pipeline calls to phantom services to prevent execution crashes.
    """
    def __getattr__(self, item: str):
        # Dynamically determine the execution context based on method naming conventions.
        # This prevents `TypeError: a coroutine was expected, got None` during asyncio.create_task() calls.
        if "async" in item.lower():
            async def _async_noop(*args, **kwargs):
                pass
            return _async_noop
        return lambda *args, **kwargs: None

class FactoryManager:
    """
    Unified singleton dependency container.
    Thread-safe registry for managing application-wide lifecycle singletons and core services.
    """
    _instance = None
    _lock = threading.Lock()
    _instances = {}

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(FactoryManager, cls).__new__(cls)
                cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if getattr(self, "_initialized", False):
            return
        self._initialized = True

    @classmethod
    def register_instance(cls, name: str, instance: Any) -> None:
        """Rapidly map a running process instance into the storage dictionary."""
        with cls._lock:
            cls._instances[name] = instance
            logger.info("[FactoryManager] Core service '%s' successfully registered in injection map.", name)

    @classmethod
    def get_instance(cls, name: str) -> Optional[Any]:
        """Fetch active singletons from the data dictionary using safe key checking."""
        with cls._lock:
            instance = cls._instances.get(name)
            
            # Defensive Pipeline Stubbing for the Phantom PredictiveHistorian Service
            if instance is None and name == "PredictiveHistorian":
                logger.warning("[FactoryManager] Un-implemented phantom service '%s' requested. Deploying SafeServiceStub.", name)
                return SafeServiceStub()
                
            return instance

    def boot_singletons(self):
        """Lifecycle marker routine for booting registered singletons."""
        logger.info("[FactoryManager] Executing global singleton boot sequence.")
        # Future expansions will iterate through registered components and invoke native boot methods.

    def shutdown_singletons(self):
        """Lifecycle marker routine to clear memory arrays and prevent stale process leaks."""
        logger.info("[FactoryManager] Initiating shutdown sequence. Evacuating registry map.")
        with self._lock:
            self._instances.clear()
