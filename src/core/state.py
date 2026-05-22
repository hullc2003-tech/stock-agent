from typing import TypedDict, List, Dict, Any, Optional
from datetime import datetime


class AgentPerformance(TypedDict):
    agent_name: str
    accuracy: float
    total_predictions: int
    correct_predictions: int
    last_updated: str


class PredictionResult(TypedDict):
    ticker: str
    prediction: str  # "UP_10%" or "NOT_UP_10%"
    confidence: float
    reasoning: str
    timestamp: str
    agents_used: List[str]


class StockAgentState(TypedDict):
    # Input
    ticker: str
    
    # Research outputs
    technical_analysis: Optional[Dict[str, Any]]
    sentiment_analysis: Optional[Dict[str, Any]]
    news_catalyst: Optional[Dict[str, Any]]
    
    # Aggregated
    combined_signal: Optional[Dict[str, Any]]
    final_prediction: Optional[PredictionResult]
    
    # Self-improvement / Learning
    performance_history: List[AgentPerformance]
    suggested_improvements: List[str]
    code_changes_applied: List[str]
    
    # Metadata
    run_id: str
    start_time: str
    errors: List[str]
    accuracy_met: bool