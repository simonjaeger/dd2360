@ECHO OFF
nvcc -O3 -lOpenCL -arch=sm_50 exercise_1.c -o exercise_1.exe || goto :error
exercise_1.exe

:error