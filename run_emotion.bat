@echo off
cd /d "%~dp0"
call deepface_env\Scripts\activate
python main.py
pause
