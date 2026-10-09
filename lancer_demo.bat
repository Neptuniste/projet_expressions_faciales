@echo off
REM Lance la demo webcam. On peut ajouter des options, par exemple : lancer_demo.bat --camera 1
cd /d "%~dp0"
.venv\Scripts\python demo_webcam.py %*
pause
