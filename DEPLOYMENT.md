# Google Cloud Run Deployment Guide for AI Stock Agent

This repository (`stock-agent`) is already well-prepared for Google Cloud with a production `Dockerfile` and `cloudbuild.yaml`. This guide provides **complete step-by-step instructions** for deploying via the Google Cloud Console (easiest) or GitHub Actions.

## Current State of the Repo

- `Dockerfile` — Production-ready (Python 3.12-slim, `src.api:app`, proper PORT handling)
- `cloudbuild.yaml` — Pre-configured for Cloud Build + Cloud Run deployment
- `main.py` / `src/` — FastAPI application with multi-agent system

## Recommended Deployment: Cloud Run + Cloud Build (via Console)

### Why Cloud Run?
- Serverless, scales automatically (including to zero)
- Excellent for FastAPI + LangGraph / multi-agent workloads
- Pay-per-use pricing
- Continuous deployment from GitHub

### Step-by-Step: Deploy from GitHub using Cloud Console

1. Go to the [Google Cloud Console](https://console.cloud.google.com/)
2. Select your project (billing must be enabled)
3. Enable required APIs (run this in Cloud Shell):
   ```bash
   gcloud services enable run.googleapis.com cloudbuild.googleapis.com artifactregistry.googleapis.com
   ```

4. Go to **Cloud Run** → **Create Service**
5. Select **Continuously deploy from a repository**
6. Click **SET UP CLOUD BUILD**
7. Choose **GitHub** as provider and authenticate
8. Select this repository: `hullc2003-tech/stock-agent`
9. Configure the trigger:
   - Branch: `^main$`
   - Build Type: **Dockerfile** (recommended — already present in repo)
   - Build context: `/`
10. Service settings:
    - Name: `stock-agent`
    - Region: `us-central1` (matches your cloudbuild.yaml)
    - Authentication: Allow unauthenticated (or configure IAM)
11. In **Advanced** settings (recommended):
    - Memory: 4Gi (or more)
    - CPU: 2
    - Min instances: 0 or 1
    - Max instances: 10–100
    - Timeout: 600s
    - Add any required environment variables or Secret Manager secrets
12. Click **Create**

Cloud Build will build the image using your existing `Dockerfile`, push to Artifact Registry, and deploy to Cloud Run.

**Every push to `main`** will automatically trigger a new revision.

## Using the Existing `cloudbuild.yaml`

Your repo already contains a `cloudbuild.yaml` configured for:
- Building the Docker image
- Pushing to Artifact Registry (`us-central1-docker.pkg.dev/$PROJECT_ID/stock-agent/...`)
- Deploying to Cloud Run service `stock-agent`

You can trigger builds manually or create a Cloud Build trigger pointing to this file.

## Alternative: GitHub Actions CI/CD

See `.github/workflows/deploy-to-cloud-run.yml` for a fully automated GitHub Actions workflow using the official `google-github-actions/deploy-cloudrun` action.

This is great if you prefer everything defined in your repo or want to add tests before deployment.

## Production Recommendations

- **Secrets**: Use Google Secret Manager for API keys (OpenAI, stock data providers, etc.) and reference them in Cloud Run
- **Resources**: Your `cloudbuild.yaml` already sets good defaults (4Gi / 2 CPU). Adjust based on agent load
- **Self-improving agents**: For persistent state, integrate Firestore, Cloud SQL, or Redis (Memorystore)
- **Monitoring**: Enable Cloud Logging + Cloud Monitoring + set up alerts
- **Custom Domain**: Map a domain in Cloud Run settings
- **Security**: Consider Cloud Armor and proper IAM policies

## Quick Local Test

```bash
pip install -r requirements.txt
uvicorn src.api:app --reload
```

## Need Help?

Open an issue or ask for specific help with:
- Setting up Workload Identity Federation
- Adding Secret Manager integration
- Scaling / cost optimization
- Database integration for agent memory
