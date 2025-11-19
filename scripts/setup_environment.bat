@echo off
REM Automated setup script for Nemotron CPT Pipeline (Windows)

echo =========================================
echo Nemotron CPT Pipeline - Environment Setup
echo =========================================
echo.

REM Check Python version
echo [1/6] Checking Python version...
python --version
python -c "import sys; exit(0 if sys.version_info >= (3, 10) else 1)"
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Python 3.10+ is required
    exit /b 1
)

REM Create virtual environment
echo.
echo [2/6] Creating virtual environment...
if not exist "venv\" (
    python -m venv venv
    echo Virtual environment created
) else (
    echo Virtual environment already exists
)

REM Activate virtual environment
echo.
echo [3/6] Activating virtual environment...
call venv\Scripts\activate.bat

REM Upgrade pip
echo.
echo [4/6] Upgrading pip...
python -m pip install --upgrade pip

REM Install dependencies
echo.
echo [5/6] Installing dependencies...
pip install -r requirements.txt

REM Install Flash Attention (optional)
echo.
echo [6/6] Installing Flash Attention (optional)...
set /p install_flash="Install Flash Attention 2? (requires compilation, y/n): "
if /i "%install_flash%"=="y" (
    pip install flash-attn --no-build-isolation
    echo Flash Attention installed
) else (
    echo Skipping Flash Attention installation
)

REM Create sample data
echo.
echo Creating sample data...
python scripts\prepare_data.py --create-sample

REM Complete
echo.
echo =========================================
echo Setup Complete!
echo =========================================
echo.
echo Next steps:
echo 1. Activate environment: venv\Scripts\activate
echo 2. Login to HuggingFace: huggingface-cli login
echo 3. Add your data to: data\custom_corpus\
echo 4. Run training: scripts\run_prototype.bat
echo.
echo For more information, see README.md and QUICKSTART.md
echo.
pause
