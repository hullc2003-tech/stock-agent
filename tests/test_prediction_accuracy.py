import pytest
import asyncio
from datetime import datetime, timedelta
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.graphs.orchestrator import run_prediction


# Historical test cases with known outcomes (simplified for initial testing)
# Format: (ticker, actual_outcome_10pct_in_24h)
# These are illustrative - in production you'd use proper backtesting data
TEST_CASES = [
    ("NVDA", True),   # Strong recent momentum example
    ("AAPL", False),  # More stable stock
    ("TSLA", True),   # Volatile - often has big moves
    ("MSFT", False),
    ("AMD", True),
]


def calculate_accuracy(results: list) -> float:
    """Calculate accuracy from list of (prediction, actual) tuples."""
    if not results:
        return 0.0
    
    correct = 0
    for pred, actual in results:
        # Consider prediction correct if model predicted UP_10% and it was actually true
        # or predicted NOT and it was false
        predicted_up = pred.get("prediction") == "UP_10%"
        if predicted_up == actual:
            correct += 1
    return correct / len(results)


@pytest.mark.asyncio
async def test_overall_accuracy_threshold():
    """
    Test that the multi-agent system achieves at least 52% accuracy
    on a small set of historical test cases.
    
    This is a gate requirement from the original specification.
    """
    results = []
    
    for ticker, actual_outcome in TEST_CASES:
        try:
            prediction_result = await run_prediction(ticker)
            final_pred = prediction_result.get("final_prediction", {})
            
            results.append((final_pred, actual_outcome))
            
            print(f"{ticker}: Predicted={final_pred.get('prediction')}, "
                  f"Confidence={final_pred.get('confidence', 0):.2f}, Actual={actual_outcome}")
        except Exception as e:
            print(f"Error predicting {ticker}: {e}")
            results.append(({"prediction": "ERROR"}, actual_outcome))
    
    accuracy = calculate_accuracy(results)
    print(f"\n=== Overall Accuracy: {accuracy:.1%} ===")
    
    # Gate requirement: >= 52% accuracy
    assert accuracy >= 0.52, (
        f"System accuracy {accuracy:.1%} is below the required 52% threshold. "
        f"The agency must reach >=52% accuracy before live use."
    )


def test_accuracy_calculation_logic():
    """Test the accuracy calculation helper."""
    test_results = [
        ({"prediction": "UP_10%"}, True),
        ({"prediction": "UP_10%"}, False),
        ({"prediction": "NOT_UP_10%"}, False),
        ({"prediction": "UP_10%"}, True),
    ]
    acc = calculate_accuracy(test_results)
    assert acc == 0.75  # 3 out of 4 correct


if __name__ == "__main__":
    # Allow running directly for quick checks
    asyncio.run(test_overall_accuracy_threshold())