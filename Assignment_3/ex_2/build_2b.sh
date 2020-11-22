export PATH=/usr/local/cuda/bin:$PATH
nvcc -O3 -arch=sm_50 exercise_2b.cu -o exercise_2b.out
./exercise_2b.out