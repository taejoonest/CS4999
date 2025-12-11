#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>

#define BLOCKSIZE 32

__global__ void sgemm_kernel2(int M, int N, int K,
                              float alpha, const float *A,
                              const float *B, float beta, float *C)
{

    unsigned int tid = threadIdx.x;
    unsigned int lane = tid % BLOCKSIZE;
    unsigned int wf = tid / BLOCKSIZE;

    unsigned int row = blockIdx.y * BLOCKSIZE + wf;
    unsigned int col = blockIdx.x * BLOCKSIZE + lane;

    if (row < M && col < N)
    {
        float tmp = 0.0f;
        for (int i = 0; i < K; ++i)
            tmp += A[row * K + i] * B[i * N + col];
        C[row * N + col] = alpha * tmp + beta * C[row * N + col];
    }
}

int main(int argc, char **argv)
{
    if (argc < 4)
    {
        printf("usage: %s M N K\n", argv[0]);
        return 1;
    }

    int M = atoi(argv[1]);
    int N = atoi(argv[2]);
    int K = atoi(argv[3]);

    size_t bytesA = (size_t)M * K * sizeof(float);
    size_t bytesB = (size_t)K * N * sizeof(float);
    size_t bytesC = (size_t)M * N * sizeof(float);

    float *A, *B, *C;
    float *d_A, *d_B, *d_C;

    A = (float *)malloc(bytesA);
    B = (float *)malloc(bytesB);
    C = (float *)malloc(bytesC);

    hipMalloc(&d_A, bytesA);
    hipMalloc(&d_B, bytesB);
    hipMalloc(&d_C, bytesC);

    for (int i = 0; i < M * K; i++)
        A[i] = 1.0f;
    for (int i = 0; i < K * N; i++)
        B[i] = 1.0f;
    for (int i = 0; i < M * N; i++)
        C[i] = 0.0f;

    hipMemcpy(d_A, A, bytesA, hipMemcpyHostToDevice);
    hipMemcpy(d_B, B, bytesB, hipMemcpyHostToDevice);
    hipMemcpy(d_C, C, bytesC, hipMemcpyHostToDevice);

    dim3 grid((N + BLOCKSIZE - 1) / BLOCKSIZE,
              (M + BLOCKSIZE - 1) / BLOCKSIZE);
    dim3 block(BLOCKSIZE * BLOCKSIZE);

    printf("Grid: (%d, %d), Block: %d threads\n", grid.x, grid.y, block.x);

    hipLaunchKernelGGL(sgemm_kernel2, grid, block, 0, 0,
                       M, N, K, 1.0f, d_A, d_B, 0.0f, d_C);
    hipDeviceSynchronize();

    hipEvent_t start, stop;
    hipEventCreate(&start);
    hipEventCreate(&stop);

    hipEventRecord(start, 0);
    for (int run = 0; run < 10; run++)
    {
        hipLaunchKernelGGL(sgemm_kernel2, grid, block, 0, 0,
                           M, N, K, 1.0f, d_A, d_B, 0.0f, d_C);
    }
    hipEventRecord(stop, 0);
    hipEventSynchronize(stop);

    float ms = 0.0f;
    hipEventElapsedTime(&ms, start, stop);
    float avg_ms = ms / 10.0f;

    double flops = 2.0 * M * N * K;
    double gflops = (flops / (avg_ms / 1000.0)) / 1e9;

    printf("Kernel 2 time: %.3f ms\n", avg_ms);
    printf("Performance: %.2f GFLOPS\n", gflops);

    hipEventDestroy(start);
    hipEventDestroy(stop);

    hipMemcpy(C, d_C, bytesC, hipMemcpyDeviceToHost);

    double sum = 0.0;
    for (int i = 0; i < M * N; i++)
    {
        sum += C[i];
    }
    printf("Sum of all C entries: %.6f\n", sum);

    hipFree(d_A);
    hipFree(d_B);
    hipFree(d_C);
    free(A);
    free(B);
    free(C);

    return 0;
}
