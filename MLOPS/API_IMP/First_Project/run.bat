@echo off

echo Starting FastAPI...
start cmd /k "uvicorn API.main:app --reload"

timeout /t 3 > nul

echo Starting Streamlit...
start cmd /k "streamlit run streamlit/app.py"

echo Both servers are running.
pause