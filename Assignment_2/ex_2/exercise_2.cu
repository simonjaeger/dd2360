#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <limits>
#include <sys/time.h>

#define ARRAY_SIZE 10000
#define BLOCK_SIZE 256

#define MICROSECONDS(start, end) ((end.tv_sec - start.tv_sec) * 1000000LL + end.tv_usec - start.tv_usec)
#define MILLISECONDS(start, end) MICROSECONDS(start, end) / 1000.0
#define SECONDS(start, end) MILLISECONDS(start, end) / 1000.0

void cpu_saxpy(const float *x, float *y, const float a)
{
    for (unsigned int i = 0; i < ARRAY_SIZE; i++)
    {
        y[i] += a * x[i];
    }
}

__global__ void gpu_saxpy(const float *x, float *y, const float a)
{
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < ARRAY_SIZE)
    {
        y[i] += a * x[i];
    }
}

double cpuSecond()
{
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return ((double)tp.tv_sec + (double)tp.tv_usec * 1.e-6);
}

int main(int argc, char **argv)
{
    // Initialize data.
    struct timeval start, end;
    float *x = (float *)malloc(ARRAY_SIZE * sizeof(float));
    float *y = (float *)malloc(ARRAY_SIZE * sizeof(float));
    const float a = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);

    for (unsigned int i = 0; i < ARRAY_SIZE; i++)
    {
        x[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
        y[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
    }

    // Copy data for CPU.
    float *cpu_x = (float *)malloc(ARRAY_SIZE * sizeof(float));
    float *cpu_y = (float *)malloc(ARRAY_SIZE * sizeof(float));
    memcpy(cpu_x, x, ARRAY_SIZE);
    memcpy(cpu_y, y, ARRAY_SIZE);

    // Run CPU SAXPY.
    printf("Computing SAXPY on the CPU... ");
    gettimeofday(&start, NULL);
    cpu_saxpy(cpu_x, cpu_y, a);
    gettimeofday(&end, NULL);
    printf("Done! Took %lfms.\n", MILLISECONDS(start, end));

    // Copy data for GPU.
    float *gpu_x = (float *)malloc(ARRAY_SIZE * sizeof(float));
    float *gpu_y = (float *)malloc(ARRAY_SIZE * sizeof(float));
    memcpy(gpu_x, x, ARRAY_SIZE);
    memcpy(gpu_y, y, ARRAY_SIZE);

    float *cuda_x;
    float *cuda_y;
    cudaMalloc(&cuda_x, ARRAY_SIZE * sizeof(float));
    cudaMalloc(&cuda_y, ARRAY_SIZE * sizeof(float));
    cudaMemcpy(cuda_x, gpu_x, ARRAY_SIZE * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(cuda_y, gpu_y, ARRAY_SIZE * sizeof(float), cudaMemcpyHostToDevice);

    // Run GPU SAXPY.
    printf("Computing SAXPY on the GPU... ");

    // Make sure that the grid size is enough to fit all elements.
    gettimeofday(&start, NULL);
    gpu_saxpy<<<(ARRAY_SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(cuda_x, cuda_y, a);
    cudaDeviceSynchronize();
    gettimeofday(&end, NULL);
    printf("Done! Took %lfms.\n", MILLISECONDS(start, end));

    cudaMemcpy(gpu_x, cuda_x, ARRAY_SIZE * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(gpu_y, cuda_y, ARRAY_SIZE * sizeof(float), cudaMemcpyDeviceToHost);

    // Compare CPU/GPU SAXPY.
    bool success = true;
    float epsilon = std::numeric_limits<float>::epsilon();
    printf("Comparing the output for each implementation... ");
    for (unsigned int i = 0; i < 25; i++)
    {
        if (abs(cpu_y[i] - gpu_y[i]) > epsilon)
        {
            success = false;
            break;
        }
    }
    printf(success ? "Correct!\n" : "Incorrect!\n");

    // Free resources.
    free(x);
    free(y);
    free(cpu_x);
    free(cpu_y);
    free(gpu_x);
    free(gpu_y);
    cudaFree(x);
    cudaFree(y);

    return 0;
}