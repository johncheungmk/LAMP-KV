@echo off
REM LAMP-KV Windows experiment runner.
REM Change PYTHON_EXE if your Python path is different.
set PYTHON_EXE=C:\Users\User\AppData\Local\Programs\Python\Python313\python.exe
set OUTDIR=results
if not exist %OUTDIR% mkdir %OUTDIR%

%PYTHON_EXE% scripts\lamp_kv_policy_test.py --pressure 0.0 --seq-len 1024 --seed 7 --out %OUTDIR%\pressure_0_0_seq1024.json
%PYTHON_EXE% scripts\lamp_kv_policy_test.py --pressure 0.5 --seq-len 1024 --seed 7 --out %OUTDIR%\pressure_0_5_seq1024.json
%PYTHON_EXE% scripts\lamp_kv_policy_test.py --pressure 1.0 --seq-len 1024 --seed 7 --out %OUTDIR%\pressure_1_0_seq1024.json

%PYTHON_EXE% scripts\lamp_kv_policy_test.py --pressure 0.5 --seq-len 512 --seed 7 --out %OUTDIR%\pressure_0_5_seq512.json

echo Done. Results written to %OUTDIR%.
