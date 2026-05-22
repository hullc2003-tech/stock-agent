import time
from functools import wraps
from typing import Callable, Any, Optional
from datetime import datetime, timedelta
import os


class SimpleRateLimiter:
    """Lightweight rate limiter suitable for free tier APIs (yfinance, Tavily, etc.).
    
    Uses in-memory tracking + configurable delay between calls.
    Good enough for single-instance deployments on free tiers.
    """
    
    def __init__(self, min_delay_seconds: float = 1.5, name: str = "default"):
        self.min_delay = min_delay_seconds
        self.name = name
        self.last_call_time: Optional[float] = None
        self.call_count = 0
    
    def wait_if_needed(self):
        """Block until enough time has passed since last call."""
        if self.last_call_time is None:
            self.last_call_time = time.time()
            return
        
        elapsed = time.time() - self.last_call_time
        if elapsed < self.min_delay:
            sleep_time = self.min_delay - elapsed
            print(f"[RateLimiter:{self.name}] Throttling: sleeping {sleep_time:.2f}s (free tier protection)")
            time.sleep(sleep_time)
        
        self.last_call_time = time.time()
        self.call_count += 1
    
    def reset(self):
        self.last_call_time = None
        self.call_count = 0


# Pre-configured limiters for common free tier APIs
YFINANCE_LIMITER = SimpleRateLimiter(min_delay_seconds=1.8, name="yfinance")
TAVILY_LIMITER = SimpleRateLimiter(min_delay_seconds=1.2, name="tavily")


def rate_limited(limiter: SimpleRateLimiter):
    """Decorator to automatically throttle function calls."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            limiter.wait_if_needed()
            return func(*args, **kwargs)
        return wrapper
    return decorator


def get_yfinance_delay() -> float:
    """Get yfinance delay from environment or use safe default."""
    return float(os.getenv("YFINANCE_DELAY_SECONDS", "1.8"))


def get_tavily_delay() -> float:
    """Get Tavily delay from environment or use safe default."""
    return float(os.getenv("TAVILY_DELAY_SECONDS", "1.2"))