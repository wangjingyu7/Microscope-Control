@echo off
Powershell.exe -executionpolicy bypass -Command ". .\base.ps1; Activate-Anaconda; python make_configuration_single.py --help"
