export PATH=/usr/local/cuda/bin:$PATH
nvcc -O3 -arch=sm_50 exercise_3.cu -o exercise_3.out --default-stream per-thread
./exercise_3.out