    initial_state: StockAgentState = {
        "ticker": ticker.upper(),
        "technical_analysis": None,
        "sentiment_analysis": None,
        "news_catalyst": None,
        "combined_signal": None,
        "final_prediction": None,
        # List fields initialized as empty lists for reducer compatibility
        "errors": [],
        "performance_history": [],
        "suggested_improvements": [],
        "code_changes_applied": [],
        "run_id": str(uuid.uuid4()),
        "start_time": datetime.utcnow().isoformat(),
        "accuracy_met": False,
    }