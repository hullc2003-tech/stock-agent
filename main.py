"""
Production entrypoint for the AI Stock Agent system.
"""

import asyncio
from src.api import app
from src.graphs.orchestrator import run_prediction

if __name__ == "__main__":
    print("AI Stock Agent v2.5 starting...")
    result = asyncio.run(run_prediction("NVDA"))
    print(result)