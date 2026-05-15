import os
from concurrent.futures import ProcessPoolExecutor

# Global ProcessPoolExecutor for CPU-heavy tasks
# Initialized lazily to avoid issues during module import in some environments
_cpu_executor = None

def get_cpu_executor():
    global _cpu_executor
    if _cpu_executor is None:
        _cpu_executor = ProcessPoolExecutor(max_workers=min(os.cpu_count() or 1, 4))
    return _cpu_executor
