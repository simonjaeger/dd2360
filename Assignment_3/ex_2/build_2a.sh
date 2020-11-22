export PATH=/usr/local/cuda/bin:$PATH
nvcc -O3 -arch=sm_50 exercise_2a.cu -o exercise_2a.out
./exercise_2a.out