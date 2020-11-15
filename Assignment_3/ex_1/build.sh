export PATH=/usr/local/cuda/bin:$PATH
nvcc -O3 -arch=sm_50 hw3_ex1.cu -o hw3_ex1.out
#srun -n 1 ./hw3_ex1.out images/hw3.bmp
./hw3_ex1.out images/hw3.bmp