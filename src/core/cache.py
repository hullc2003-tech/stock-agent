import json
import os
import time
from typing import Any, Optional, Callable
from functools import wraps

CACHE_DIR = "cache"
os.makedirs(CACHE_DIR, exist_ok=True)


def get_cache_path(key: str) -> str:
    safe_key = key.replace("/", "_").replace(":", "_")
    return os.path.join(CACHE_DIR, f"{safe_key}.json")


def get_from_cache(key: str, ttl_seconds: int = 3600) -> Optional[Any]:
    path = get_cache_path(key)
    if not os.path.exists(path):
        return None
    
    try:
        with open(path, "r") as f:
            data = json.load(f)
        
        if time.time() - data.get("timestamp", 0) > ttl_seconds:
            return None  # Expired
        return data.get("value")
    except:
        return None


def save_to_cache(key: str, value: Any):
    path = get_cache_path(key)
    try:
        with open(path, "w") as f:
            json.dump({
                "timestamp": time.time(),
                "value": value
            }, f)
    except Exception as e:
        print(f"Cache write error: {e}")


def cached(ttl_seconds: int = 3600):
    """Decorator for simple file-based caching with TTL. Good for free tier."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Create a simple cache key from function name + first arg (usually ticker)
            cache_key = f"{func.__name__}_{args[0] if args else 'default'}"
            cached_result = get_from_cache(cache_key, ttl_seconds)
            if cached_result is not None:
                print(f"[Cache] Hit for {cache_key}")
                return cached_result
            
            result = func(*args, **kwargs)
            save_to_cache(cache_key, result)
            return result
        return wrapper
    return decorator