#include <stdio.h>

__global__ void cuda_printf()
{
    const unsigned int threadId = threadIdx.x;
    printf("Hello World! My threadId is %d\n", threadId);
}

int main(int argc, char **argv)
{
    dim3 grid(1);
    dim3 block(256);

    cuda_printf<<<grid, block>>>();

    cudaDeviceSynchronize();

    return 0;
}