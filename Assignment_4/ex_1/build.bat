@ECHO OFF
nvcc -O3 -lOpenCL -arch=sm_50 exercise_1.c -o exercise_1.exe
exercise_1.exe
