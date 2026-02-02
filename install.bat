@echo off
TITLE ChitraGupta - Installation
echo ============================================================
echo   ChitraGupta - One-Time Setup
echo ============================================================
echo.
echo Installing Flask...
pip install flask
echo.
echo Installing llama-cpp-python with CUDA support (RTX 4050)...
pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124
echo.
echo ============================================================
echo   Installation complete!
echo   Run 'run.bat' to start ChitraGupta
echo ============================================================
pause
