@echo off
set PYTHONPATH=%~dp0
echo PYTHONPATH set to %PYTHONPATH%
python tools/run_net.py %*
