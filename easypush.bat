echo off
set arg1=%1
shift
git add -A
git commit -m "%arg1%"
git pull origin main
git push origin main 