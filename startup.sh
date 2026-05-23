#!/bin/bash
cd /home/site/wwwroot
export PYTHONPATH=/home/site/wwwroot:$PYTHONPATH
pip install -r requirements.txt
gunicorn -k uvicorn.workers.UvicornWorker -b 0.0.0.0:$PORT main:app
