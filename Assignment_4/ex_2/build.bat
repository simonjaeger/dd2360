@ECHO OFF
nvcc -O3 -lOpenCL -arch=sm_50 exercise_2.c -o exercise_2.exe || goto :error
exercise_2.exe

:error