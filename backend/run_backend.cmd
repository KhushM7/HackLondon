@echo off
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0run_backend.ps1" %*
