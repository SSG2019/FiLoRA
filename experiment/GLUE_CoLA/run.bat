@echo off
setlocal

if not exist log mkdir log
set PYTHONHASHSEED=0

for /L %%s in (0,1,4) do (
  echo Running seed=%%s
  set SEED=%%s
  call python train.py > log\seed_%%s.log 2>&1
)

endlocal