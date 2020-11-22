#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <limits>
#include <sys/time.h>

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

    // Initialize data.
    Particle *particles;
    cudaMallocManaged(&particles, NUM_PARTICLES * sizeof(Particle));

    for (unsigned int i = 0; i < NUM_PARTICLES; i++)
    {
        particles[i].position.x = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
        particles[i].position.y = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
        particles[i].position.z = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
        particles[i].velocity.x = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
        particles[i].velocity.y = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
        particles[i].velocity.z = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
    }

    // Initialize data.
    Particle *gpu_particles;
    cudaMalloc(&gpu_particles, NUM_PARTICLES * sizeof(Particle));

    // Run simulation.
    printf("Running simulation... ");
    gettimeofday(&start, NULL);
    for (unsigned int i = 0; i < NUM_ITERATIONS; i++)
    {
        gpu_timestep<<<(NUM_PARTICLES + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(gpu_particles, dt);
        cudaDeviceSynchronize();
    }
    gettimeofday(&end, NULL);
    printf("Done! Took %lfs.\n", SECONDS(start, end));

    // Free resources.
    cudaFree(particles);

    return 0;
}