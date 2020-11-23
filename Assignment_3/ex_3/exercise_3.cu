#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <limits>
#include <sys/time.h>

#define NUM_PARTICLES 1000000
#define NUM_ITERATIONS 1000
#define NUM_BATCHES 4 // MUST BE A DIVIDER OF NUM_PARTICLES
#define BLOCK_SIZE 200

#define STREAMS

#define MICROSECONDS(start, end) ((end.tv_sec - start.tv_sec) * 1000000LL + end.tv_usec - start.tv_usec)
#define MILLISECONDS(start, end) MICROSECONDS(start, end) / 1000.0
#define SECONDS(start, end) MILLISECONDS(start, end) / 1000.0

typedef struct
{
    float3 position;
    float3 velocity;
} Particle;

void cpu_timestep(Particle *particles, const float dt)
{
    for (unsigned int i = 0; i < NUM_PARTICLES; i++)
    {
        particles[i].position.x += particles[i].velocity.x * dt;
        particles[i].position.y += particles[i].velocity.y * dt;
        particles[i].position.z += particles[i].velocity.z * dt;
    }
}

__global__ void gpu_timestep(Particle *particles, const float dt)
{
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < NUM_PARTICLES)
    {
        particles[i].position.x += particles[i].velocity.x * dt;
        particles[i].position.y += particles[i].velocity.y * dt;
        particles[i].position.z += particles[i].velocity.z * dt;
    }
}

int main(int argc, char **argv)
{
#ifdef STREAMS
    const int streamSize = NUM_PARTICLES / NUM_BATCHES;
    const int streamBytes = streamSize * sizeof(Particle);
    cudaStream_t stream[NUM_BATCHES];

    for (unsigned int i = 0; i < NUM_BATCHES; i++)
    {
        cudaStreamCreate(&stream[i]);
    }
#endif

    struct timeval start, end;
    const float dt = 1.0;

    // Initialize CPU data.
    Particle *cpu_particles;
    cudaMallocHost(&cpu_particles, NUM_PARTICLES * sizeof(Particle));

    for (unsigned int i = 0; i < NUM_PARTICLES; i++)
    {
        cpu_particles[i].position.x = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
        cpu_particles[i].position.y = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
        cpu_particles[i].position.z = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
        cpu_particles[i].velocity.x = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
        cpu_particles[i].velocity.y = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
        cpu_particles[i].velocity.z = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
    }

    // Initialize data.
    Particle *gpu_particles;
    cudaMalloc(&gpu_particles, NUM_PARTICLES * sizeof(Particle));

    // Run simulation.
    printf("Running simulation... ");
    gettimeofday(&start, NULL);
    for (unsigned int i = 0; i < NUM_ITERATIONS; i++)
    {
#ifdef STREAMS
        for (unsigned int j = 0; j < NUM_BATCHES; j++)
        {
            const int offset = j * streamSize;
            cudaMemcpyAsync(&gpu_particles[offset], &cpu_particles[offset], streamBytes, cudaMemcpyHostToDevice, stream[j]);
            gpu_timestep<<<streamSize / BLOCK_SIZE, BLOCK_SIZE, 0, stream[j]>>>(&gpu_particles[offset], dt);
            cudaMemcpyAsync(&cpu_particles[offset], &gpu_particles[offset], streamBytes, cudaMemcpyDeviceToHost, stream[j]);
        }
        cudaDeviceSynchronize();
#else
        cudaMemcpy(gpu_particles, cpu_particles, NUM_PARTICLES * sizeof(Particle), cudaMemcpyHostToDevice);
        gpu_timestep<<<(NUM_PARTICLES + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(gpu_particles, dt);
        cudaDeviceSynchronize();
        cudaMemcpy(cpu_particles, gpu_particles, NUM_PARTICLES * sizeof(Particle), cudaMemcpyDeviceToHost); // Copy anywhere.
#endif
    }

    // Uncomment to check that values are updated as they should.
    // for (unsigned int i = 0; i < 10; i++)
    // {
    //     printf("%f,%f,%f\n", cpu_particles[i].position.x, cpu_particles[i].position.y, cpu_particles[i].position.z);
    // } 

    gettimeofday(&end, NULL);
    printf("Done! Took %lfs.\n", SECONDS(start, end));

    // Free resources.
    cudaFreeHost(cpu_particles);
    cudaFree(gpu_particles);
    
#ifdef STREAMS
    for (unsigned int i = 0; i < NUM_BATCHES; i++)
    {
        cudaStreamDestroy(stream[i]);
    }
#endif

    return 0;
}