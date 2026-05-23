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
- Production FastAPI ready
- Rate limiting + error handling

## Quick Start (Local)
```bash
pip install -r requirements.txt
export OPENAI_API_KEY=sk-...
uvicorn src.api:app --reload
```

## Deployment

### Google Cloud Run (Recommended)

This repo is fully prepared for Google Cloud Run.

**Easiest method**: Use the Google Cloud Console → "Continuously deploy from a repository" (see [DEPLOYMENT.md](DEPLOYMENT.md) for full steps).

Your repository already includes:
- Production `Dockerfile`
- Pre-configured `cloudbuild.yaml` for Cloud Build + Cloud Run

**Alternative**: GitHub Actions workflow in `.github/workflows/deploy-to-cloud-run.yml`

Full instructions, production recommendations, and troubleshooting are in **[DEPLOYMENT.md](DEPLOYMENT.md)**.

---

See full deployment guide in the conversation history or ask me for Azure steps if needed.