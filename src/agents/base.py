from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from datetime import datetime
import json
import os

from src.core.config import settings


class BaseAgent(ABC):
    """Base class for all agents with self-improvement capabilities."""
    
    def __init__(self, name: str):
        self.name = name
        self.performance_log_path = settings.performance_log_path
        os.makedirs(os.path.dirname(self.performance_log_path), exist_ok=True)
    
    @abstractmethod
    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the agent's main logic. Must be implemented by subclasses."""
        pass
    
    def log_performance(self, accuracy: float, total: int, correct: int):
        """Log agent performance for self-improvement analysis."""
        entry = {
            "agent_name": self.name,
            "accuracy": accuracy,
            "total_predictions": total,
            "correct_predictions": correct,
            "timestamp": datetime.utcnow().isoformat()
        }
        with open(self.performance_log_path, "a") as f:
            f.write(json.dumps(entry) + "\n")
    
    def get_recent_performance(self, limit: int = 20) -> List[Dict]:
        """Read recent performance logs for self-reflection."""
        if not os.path.exists(self.performance_log_path):
            return []
        
        performances = []
        with open(self.performance_log_path, "r") as f:
            for line in f.readlines()[-limit:]:
                try:
                    performances.append(json.loads(line.strip()))
                except:
                    continue
        return performances
    
    def suggest_improvements(self, recent_performance: List[Dict]) -> List[str]:
        """Basic self-reflection logic. Subclasses can override for smarter suggestions."""
        suggestions = []
        if not recent_performance:
            return suggestions
        
        avg_accuracy = sum(p["accuracy"] for p in recent_performance) / len(recent_performance)
        if avg_accuracy < settings.min_accuracy_threshold:
            suggestions.append(f"{self.name} accuracy ({avg_accuracy:.1%}) is below target {settings.min_accuracy_threshold:.0%}. Consider improving prompts, adding more data sources, or adjusting decision thresholds.")
        return suggestions