@echo off
setlocal

set "VS_ROOT=D:\Apps\Microsoft Visual Studio\2026\community"
call "%VS_ROOT%\VC\Auxiliary\Build\vcvars64.bat"
if errorlevel 1 exit /b %errorlevel%

"%VS_ROOT%\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe" %*
exit /b %errorlevel%
