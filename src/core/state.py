from typing import TypedDict, List, Dict, Any, Optional, Annotated
from datetime import datetime


def append_list(left: Optional[List], right: Optional[List]) -> List:
    """Reducer that appends lists instead of overwriting them."""
    if left is None:
        left = []
    if right is None:
        right = []
    return left + right


def append_performance(left: Optional[List], right: Optional[List]) -> List:
    """Reducer specifically for performance history."""
    if left is None:
        left = []
    if right is None:
        right = []
    return left + right


class AgentPerformance(TypedDict):
    """Record of an agent's performance for self-improvement analysis."""
    agent_name: str
    accuracy: float
    total_predictions: int
    correct_predictions: int
    last_updated: str


class PredictionResult(TypedDict):
    """Final structured prediction output."""
    ticker: str
    prediction: str  # "UP_10%" or "NOT_UP_10%"
    confidence: float
    reasoning: str
    timestamp: str
    agents_used: List[str]


class StockAgentState(TypedDict):
    """
    Central state object for the Stock Prediction Multi-Agent System.
    
    This state flows through the LangGraph workflow.
    List fields use reducers so multiple agents can safely append data.
    """

    # === Input ===
    ticker: str
    run_id: str
    start_time: str

    # === Research Outputs (one key per researcher agent) ===
    technical_analysis: Optional[Dict[str, Any]]
    sentiment_analysis: Optional[Dict[str, Any]]
    news_catalyst: Optional[Dict[str, Any]]
    # Add more researcher outputs here as you expand (e.g. fundamentals, options_flow, macro, etc.)

    # === Aggregated / Final Output ===
    combined_signal: Optional[Dict[str, Any]]
    final_prediction: Optional[PredictionResult]

    # === Self-Improvement & Learning System ===
    # These lists use reducers so multiple nodes can append safely
    errors: Annotated[List[str], append_list]
    performance_history: Annotated[List[AgentPerformance], append_performance]
    suggested_improvements: Annotated[List[str], append_list]
    code_changes_applied: Annotated[List[str], append_list]

    # === Control Flags ===
    accuracy_met: bool
