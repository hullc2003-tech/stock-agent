"""
Production entrypoint for the AI Stock Agent v2.5
Self-improving multi-agent system for ≥10% stock moves in ≤24h
"""

import asyncio
import sys
from src.graphs.orchestrator import run_prediction


async def main():
    if len(sys.argv) > 1:
        ticker = sys.argv[1]
    else:
        ticker = "NVDA"
    
    print(f"\n=== AI Stock Agent v2.5 ===")
    print(f"Running prediction for {ticker}...\n")
    
    result = await run_prediction(ticker)
    
    print("\n=== FINAL RESULT ===")
    prediction = result.get("final_prediction", {})
    print(f"Ticker: {prediction.get('ticker')}")
    print(f"Prediction: {prediction.get('prediction')}")
    print(f"Confidence: {prediction.get('confidence', 0):.1%}")
    print(f"Reasoning: {prediction.get('reasoning')}")
    print(f"\nAgents used: {prediction.get('agents_used', [])}")
    print(f"\nSuggested improvements from Learning Agent:")
    for s in result.get("suggested_improvements", []):
        print(f"  - {s}")
    
    if result.get("code_changes_applied"):
        print(f"\nCode changes applied by Code Writing Agent:")
        for c in result.get("code_changes_applied"):
            print(f"  - {c}")


if __name__ == "__main__":
    asyncio.run(main())