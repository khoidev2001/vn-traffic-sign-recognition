@echo off
cd /d %~dp0
.venv\Scripts\python -m uvicorn app.server:app --port 8000

