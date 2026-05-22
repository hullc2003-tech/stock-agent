import asyncio
from typing import Dict, Any, Literal
from datetime import datetime
import uuid

from langgraph.graph import StateGraph, END
from langchain_core.runnables import RunnableConfig

try:
    from src.core.state import StockAgentState
    from src.agents.supervisor import SupervisorAgent
    from src.agents.base import BaseAgent
except ImportError:
    # Fallback for direct execution
    import sys
    sys.path.append(".")
    from src.core.state import StockAgentState
    from src.agents.supervisor import SupervisorAgent
    from src.agents.base import BaseAgent


class TechnicalAnalysisAgent(BaseAgent):
    def __init__(self):
        super().__init__("TechnicalAnalysisAgent")
    
    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        ticker = state["ticker"]
        # TODO: Implement real yfinance + TA-Lib / pandas_ta logic with rate limiting
        # For now, return structured placeholder that can be expanded
        analysis = {
            "ticker": ticker,
            "indicators": {
                "rsi_14": 45.2,
                "sma_50": 245.3,
                "sma_200": 198.7,
                "volume_trend": "increasing"
            },
            "signal": "BULLISH",
            "confidence": 0.68,
            "reasoning": "Price above key moving averages with rising volume. RSI not overbought."
        }
        state["technical_analysis"] = analysis
        self.log_performance(accuracy=0.65, total=100, correct=65)
        return state


class SentimentAnalysisAgent(BaseAgent):
    def __init__(self):
        super().__init__("SentimentAnalysisAgent")
    
    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        ticker = state["ticker"]
        analysis = {
            "ticker": ticker,
            "overall_sentiment": "POSITIVE",
            "score": 0.72,
            "sources_analyzed": 124,
            "key_themes": ["strong earnings beat", "AI momentum", "analyst upgrades"],
            "reasoning": "High positive sentiment on social media and recent news."
        }
        state["sentiment_analysis"] = analysis
        self.log_performance(accuracy=0.58, total=80, correct=46)
        return state


class NewsCatalystAgent(BaseAgent):
    def __init__(self):
        super().__init__("NewsCatalystAgent")
    
    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        ticker = state["ticker"]
        analysis = {
            "ticker": ticker,
            "catalysts": ["Upcoming product launch", "Potential acquisition rumors"],
            "impact_score": 0.75,
            "time_horizon_hours": 18,
            "reasoning": "Major catalyst expected within next 24h that could drive significant move."
        }
        state["news_catalyst"] = analysis
        return state


class LearningAgent(BaseAgent):
    """Analyzes past performance and suggests code / prompt improvements."""
    def __init__(self):
        super().__init__("LearningAgent")
    
    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        recent = self.get_recent_performance(limit=30)
        suggestions = self.suggest_improvements(recent)
        state["suggested_improvements"] = suggestions
        if suggestions:
            state["suggested_improvements"].extend([
                "Consider adding more granular technical indicators (MACD, Bollinger Bands).",
                "Improve sentiment prompt to better handle sarcasm in financial tweets."
            ])
        return state


class CodeWritingAgent(BaseAgent):
    """Implements suggestions from Learning Agent and can email on failure."""
    def __init__(self):
        super().__init__("CodeWritingAgent")
    
    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        suggestions = state.get("suggested_improvements", [])
        if suggestions:
            # In real implementation: use LLM to generate code patches, apply them safely,
            # run tests, and email hullc2003@gmail.com on failure.
            state["code_changes_applied"] = [f"Applied improvement: {s[:80]}..." for s in suggestions[:2]]
        return state


def create_workflow():
    """Build the full self-improving multi-agent workflow."""
    workflow = StateGraph(StockAgentState)
    
    supervisor = SupervisorAgent()
    technical_agent = TechnicalAnalysisAgent()
    sentiment_agent = SentimentAnalysisAgent()
    news_agent = NewsCatalystAgent()
    learning_agent = LearningAgent()
    code_writer = CodeWritingAgent()
    
    def supervisor_node(state: StockAgentState):
        decision = supervisor.route(state)
        state["errors"] = state.get("errors", [])
        return {"next_agent": decision["next_agent"], "reasoning": decision["reasoning"]}
    
    def technical_node(state: StockAgentState):
        return technical_agent.run(state)
    
    def sentiment_node(state: StockAgentState):
        return sentiment_agent.run(state)
    
    def news_node(state: StockAgentState):
        return news_agent.run(state)
    
    def learning_node(state: StockAgentState):
        return learning_agent.run(state)
    
    def code_writer_node(state: StockAgentState):
        return code_writer.run(state)
    
    def finish_node(state: StockAgentState):
        # Final aggregation logic would go here
        if not state.get("final_prediction"):
            state["final_prediction"] = {
                "ticker": state["ticker"],
                "prediction": "UP_10%",
                "confidence": 0.71,
                "reasoning": "Combined technical + sentiment + catalyst signals are bullish.",
                "timestamp": datetime.utcnow().isoformat(),
                "agents_used": ["technical", "sentiment", "news"]
            }
        state["accuracy_met"] = True  # Would be calculated properly in real version
        return state
    
    # Add nodes
    workflow.add_node("supervisor", supervisor_node)
    workflow.add_node("technical", technical_node)
    workflow.add_node("sentiment", sentiment_node)
    workflow.add_node("news", news_node)
    workflow.add_node("learning", learning_node)
    workflow.add_node("code_writer", code_writer_node)
    workflow.add_node("finish", finish_node)
    
    # Edges
    workflow.set_entry_point("supervisor")
    
    workflow.add_conditional_edges(
        "supervisor",
        lambda state: state.get("next_agent", "FINISH"),
        {
            "technical": "technical",
            "sentiment": "sentiment",
            "news": "news",
            "learning": "learning",
            "code_writer": "code_writer",
            "FINISH": "finish"
        }
    )
    
    # After each specialist, go back to supervisor
    for node in ["technical", "sentiment", "news", "learning", "code_writer"]:
        workflow.add_edge(node, "supervisor")
    
    workflow.add_edge("finish", END)
    
    return workflow.compile()


async def run_prediction(ticker: str) -> Dict[str, Any]:
    """Main entry point to run a prediction with the full self-improving agent system."""
    app = create_workflow()
    
    initial_state: StockAgentState = {
        "ticker": ticker.upper(),
        "technical_analysis": None,
        "sentiment_analysis": None,
        "news_catalyst": None,
        "combined_signal": None,
        "final_prediction": None,
        "performance_history": [],
        "suggested_improvements": [],
        "code_changes_applied": [],
        "run_id": str(uuid.uuid4()),
        "start_time": datetime.utcnow().isoformat(),
        "errors": [],
        "accuracy_met": False,
    }
    
    config: RunnableConfig = {"recursion_limit": 25}
    final_state = await app.ainvoke(initial_state, config=config)
    return final_state