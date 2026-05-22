from typing import Dict, Any, Literal
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from src.core.config import settings
from src.core.state import StockAgentState


class RouteResponse(BaseModel):
    next_agent: Literal["technical", "sentiment", "news", "learning", "code_writer", "FINISH"] = Field(
        description="The next agent to route to or FINISH if done."
    )
    reasoning: str = Field(description="Why this agent was chosen.")


class SupervisorAgent:
    """Supervisor that intelligently routes between specialized agents and manages the self-improvement loop."""
    
    def __init__(self):
        self.llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0,
            api_key=settings.openai_api_key
        ).with_structured_output(RouteResponse)
    
    def route(self, state: StockAgentState) -> Dict[str, Any]:
        """Decide the next step based on current state."""
        system_prompt = """You are the Supervisor of a stock prediction multi-agent system.
Your goal is to predict whether a stock will rise 10% or more in the next 24 hours or less.

You have these specialized agents available:
- technical: Technical Analysis Agent (price action, indicators, volume)
- sentiment: Sentiment Analysis Agent (news, social mood)
- news: News & Catalyst Agent
- learning: Learning Agent (analyzes past performance and suggests improvements)
- code_writer: Code Writing Agent (implements improvements from Learning Agent)

Current state summary:
- Ticker: {ticker}
- Has technical analysis: {has_technical}
- Has sentiment: {has_sentiment}
- Has news: {has_news}
- Has final prediction: {has_prediction}
- Suggested improvements pending: {has_suggestions}

Decide the next agent to call. Use 'FINISH' only when you have enough information for a high-confidence prediction or when the self-improvement cycle is complete.""".format(
            ticker=state.get("ticker", "UNKNOWN"),
            has_technical=bool(state.get("technical_analysis")),
            has_sentiment=bool(state.get("sentiment_analysis")),
            has_news=bool(state.get("news_catalyst")),
            has_prediction=bool(state.get("final_prediction")),
            has_suggestions=bool(state.get("suggested_improvements")),
        )
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Current state: {state}")
        ]
        
        response: RouteResponse = self.llm.invoke(messages)
        return {
            "next_agent": response.next_agent,
            "reasoning": response.reasoning
        }
    
    def should_continue(self, state: StockAgentState) -> Literal["continue", "end"]:
        if state.get("final_prediction") and not state.get("suggested_improvements"):
            return "end"
        if len(state.get("errors", [])) > 3:
            return "end"
        return "continue"