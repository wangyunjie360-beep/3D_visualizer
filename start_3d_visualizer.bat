@echo off
setlocal

set "ENV_NAME=3d_visualizer"
set "SCRIPT_DIR=%~dp0"
set "APP=%SCRIPT_DIR%pcd_viewer_app.py"
set "CONDA_CMD="

cd /d "%SCRIPT_DIR%"

where conda >nul 2>nul
if not errorlevel 1 set "CONDA_CMD=conda"

if not defined CONDA_CMD if exist "%CONDA_PREFIX%\condabin\conda.bat" set "CONDA_CMD=%CONDA_PREFIX%\condabin\conda.bat"
if not defined CONDA_CMD if exist "%USERPROFILE%\miniconda3\condabin\conda.bat" set "CONDA_CMD=%USERPROFILE%\miniconda3\condabin\conda.bat"
if not defined CONDA_CMD if exist "%USERPROFILE%\anaconda3\condabin\conda.bat" set "CONDA_CMD=%USERPROFILE%\anaconda3\condabin\conda.bat"
if not defined CONDA_CMD if exist "D:\miniconda3\condabin\conda.bat" set "CONDA_CMD=D:\miniconda3\condabin\conda.bat"
if not defined CONDA_CMD if exist "D:\anaconda3\condabin\conda.bat" set "CONDA_CMD=D:\anaconda3\condabin\conda.bat"

if not defined CONDA_CMD (
    echo [ERROR] Conda was not found.
    echo Install Miniconda/Anaconda or add conda to PATH, then try again.
    pause
    exit /b 1
)

echo Starting 3D Visualizer with conda environment: %ENV_NAME%

if /I "%CONDA_CMD%"=="conda" (
    call conda run -n "%ENV_NAME%" python "%APP%" --device auto %*
) else (
    call "%CONDA_CMD%" run -n "%ENV_NAME%" python "%APP%" --device auto %*
)

set "EXIT_CODE=%ERRORLEVEL%"
if not "%EXIT_CODE%"=="0" (
    echo.
    echo [ERROR] 3D Visualizer failed to start. Exit code: %EXIT_CODE%
    echo Try this command from the project directory:
    echo conda run -n %ENV_NAME% python pcd_viewer_app.py --device auto
    pause
)

exit /b %EXIT_CODE%
