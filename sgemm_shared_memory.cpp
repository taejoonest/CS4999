#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>

#define BLOCKSIZE 32

__global__ void sgemm_tiled(int M, int N, int K,
                            float alpha,
                            const float *A,
                            const float *B,
                            float beta,
                            float *C)
{
  __shared__ float As[BLOCKSIZE * BLOCKSIZE];
  __shared__ float Bs[BLOCKSIZE * BLOCKSIZE];

  int threadRow = threadIdx.x / BLOCKSIZE;
  int threadCol = threadIdx.x % BLOCKSIZE;

  int cRow = blockIdx.y;
  int cCol = blockIdx.x;

  const float *Ap = A;
  const float *Bp = B;
  float *Cp = C;

  Ap += cRow * BLOCKSIZE * K;
  Bp += cCol * BLOCKSIZE;
  Cp += cRow * BLOCKSIZE * N + cCol * BLOCKSIZE;

  float tmp = 0.0f;

  for (int bkIdx = 0; bkIdx < K; bkIdx += BLOCKSIZE)
  {
    As[threadRow * BLOCKSIZE + threadCol] =
        Ap[threadRow * K + threadCol];
    Bs[threadRow * BLOCKSIZE + threadCol] =
        Bp[threadRow * N + threadCol];

    __syncthreads();

    Ap += BLOCKSIZE;
    Bp += BLOCKSIZE * N;
    for (int dotIdx = 0; dotIdx < BLOCKSIZE; ++dotIdx)
    {
      tmp += As[threadRow * BLOCKSIZE + dotIdx] *
             Bs[dotIdx * BLOCKSIZE + threadCol];
    }

    __syncthreads();
  }

  Cp[threadRow * N + threadCol] =
      alpha * tmp + beta * Cp[threadRow * N + threadCol];
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

  float *A = (float *)malloc(bytesA);
  float *B = (float *)malloc(bytesB);
  float *C = (float *)malloc(bytesC);

  for (int i = 0; i < M * K; ++i)
    A[i] = 1.0f;
  for (int i = 0; i < K * N; ++i)
    B[i] = 1.0f;
  for (int i = 0; i < M * N; ++i)
    C[i] = 0.0f;

  float *d_A, *d_B, *d_C;
  hipMalloc(&d_A, bytesA);
  hipMalloc(&d_B, bytesB);
  hipMalloc(&d_C, bytesC);

  hipMemcpy(d_A, A, bytesA, hipMemcpyHostToDevice);
  hipMemcpy(d_B, B, bytesB, hipMemcpyHostToDevice);
  hipMemcpy(d_C, C, bytesC, hipMemcpyHostToDevice);

  dim3 grid((M + BLOCKSIZE - 1) / BLOCKSIZE,
            (N + BLOCKSIZE - 1) / BLOCKSIZE);

  dim3 block(BLOCKSIZE * BLOCKSIZE);

  hipEvent_t start, stop;
  hipEventCreate(&start);
  hipEventCreate(&stop);
  hipEventRecord(start, 0);

  hipLaunchKernelGGL(sgemm_tiled, grid, block, 0, 0,
                     M, N, K, 1.0f, d_A, d_B, 0.0f, d_C);

  hipEventRecord(stop, 0);
  hipEventSynchronize(stop);

  float ms = 0.0f;
  hipEventElapsedTime(&ms, start, stop);
  printf("Tiled shared-mem kernel runtime: %.3f ms\n", ms);

  hipEventDestroy(start);
  hipEventDestroy(stop);

  hipMemcpy(C, d_C, bytesC, hipMemcpyDeviceToHost);
  hipDeviceSynchronize();

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
