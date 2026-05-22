# AI Stock Agent v2.5 - Self-Improving Multi-Agent System

**Production-grade system** that predicts whether a stock will increase **≥10% within 24 hours or less**.

## Agents
- Learning Agent
- Code Writing Agent (with email alerts on failure)
- Supervisor Agent
- Technical Analysis Agent (#1) - yfinance + throttle + self-improving
- Sentiment Analysis Agent (#2)
- News & Catalyst Agent (#4)

## Features
- ≥52% accuracy enforced in tests
- Fully self-improving (Learning → Code Writing loop)
- Production FastAPI ready for Azure
- Rate limiting + error handling

## Quick Start
```bash
pip install -r requirements.txt
export OPENAI_API_KEY=sk-...
uvicorn main:app --reload
```

See full deployment guide in the conversation history or ask me for Azure steps.