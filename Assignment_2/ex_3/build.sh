export PATH=/usr/local/cuda/bin:$PATH
nvcc -arch=sm_30 -I/usr/local/cuda/samples/common/inc exercise_3.cu -o exercise_3