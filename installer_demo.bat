@echo off
REM Installe ce qu'il faut pour la demo webcam, dans un environnement Python isole (.venv) a cote de ce fichier.
REM Il faut Python 3.11, 3.12 ou 3.13 : la version de Keras qui a enregistre le modele ne s'installe pas sous Python 3.10.
cd /d "%~dp0"
set "PY="
for %%v in (3.12 3.11 3.13) do (
  if not defined PY (
    py -%%v -c "import sys" >nul 2>&1 && set "PY=py -%%v"
  )
)
if not defined PY (
  python -c "import sys; sys.exit(0 if (3, 11) <= sys.version_info[:2] <= (3, 13) else 1)" >nul 2>&1 && set "PY=python"
)
if not defined PY (
  echo.
  echo Il faut Python 3.11 ou 3.12 : https://www.python.org/downloads/
  echo Python 3.10 ne suffit pas. On peut garder le 3.10 et installer le 3.12 a cote.
  echo.
  pause
  exit /b 1
)
echo Python utilise : %PY%
%PY% --version
if exist .venv rmdir /s /q .venv
%PY% -m venv .venv
.venv\Scripts\python -m pip install --upgrade pip
.venv\Scripts\python -m pip install -r requirements_demo.txt
echo.
echo Installation terminee. Pour lancer la demo : double-clic sur lancer_demo.bat
pause
