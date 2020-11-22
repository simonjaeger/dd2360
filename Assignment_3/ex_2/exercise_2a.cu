#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <limits>
#include <sys/time.h>

// #define PINNED_MEMORY

#define NUM_PARTICLES 100000
#define NUM_ITERATIONS 1000
#define BLOCK_SIZE 256

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
    struct timeval start, end;
    const float dt = 1.0;

    // Initialize CPU data.
#ifdef PINNED_MEMORY
    Particle *cpu_particles;
    cudaMallocHost(&cpu_particles, NUM_PARTICLES * sizeof(Particle));
#else
    Particle *cpu_particles = (Particle *)malloc(NUM_PARTICLES * sizeof(Particle));
#endif

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
        cudaMemcpy(gpu_particles, cpu_particles, NUM_PARTICLES * sizeof(Particle), cudaMemcpyHostToDevice);
        gpu_timestep<<<(NUM_PARTICLES + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(gpu_particles, dt);
        cudaDeviceSynchronize();
        cudaMemcpy(cpu_particles, gpu_particles, NUM_PARTICLES * sizeof(Particle), cudaMemcpyDeviceToHost); // Copy anywhere.
    }
    gettimeofday(&end, NULL);
    printf("Done! Took %lfs.\n", SECONDS(start, end));

    // Free resources.
#ifdef PINNED_MEMORY
    cudaFreeHost(cpu_particles);
#else
    free(cpu_particles);
#endif
    cudaFree(gpu_particles);

    return 0;
}