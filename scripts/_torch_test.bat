@echo off
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1
set CUDA_VISIBLE_DEVICES=0
cd /d D:\AI\cts
python -u scripts/_test_torch.py
echo EXIT_CODE=%ERRORLEVEL%
