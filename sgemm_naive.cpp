#include <hip/hip_runtime.h>

__global__ void sgemm_naive(int M, int N, int K, float alpha,
                            const float *A, const float *B,
                            float beta, float *C)
{
  const uint x = blockIdx.x * blockDim.x + threadIdx.x;
  const uint y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < M && y < N)
  {
    float tmp = 0.0f;
    for (int i = 0; i < K; ++i)
      tmp += A[x * K + i] * B[i * N + y];
    C[x * N + y] = alpha * tmp + beta * C[x * N + y];
  }
}

int main(int argc, char *argv[])
{
  int M = 4096, N = 4096, K = 4096;
  size_t bytesA = M * K * sizeof(float);
  size_t bytesB = K * N * sizeof(float);
  size_t bytesC = M * N * sizeof(float);

  float *A, *B, *C;
  float *d_A, *d_B, *d_C;
  hipError_t e;
  A = (float *)malloc(bytesA);
  B = (float *)malloc(bytesB);
  C = (float *)malloc(bytesC);
  for (int i = 0; i < M * K; i++)
    A[i] = 1.0f;
  for (int i = 0; i < K * N; i++)
    B[i] = 1.0f;
  for (int i = 0; i < M * N; i++)
    C[i] = 0.0f;
  e = hipMalloc(&d_A, bytesA);
  hipMalloc(&d_B, bytesB);
  hipMalloc(&d_C, bytesC);
  printf("hipMalloc: %s\n", hipGetErrorString(e));
  if (e != hipSuccess)
    return 0;
  hipMemcpy(d_A, A, bytesA, hipMemcpyHostToDevice);
  hipMemcpy(d_B, B, bytesB, hipMemcpyHostToDevice);
  hipMemcpy(d_C, C, bytesC, hipMemcpyHostToDevice);

  dim3 block(32, 32);
  dim3 grid((M + block.x - 1) / block.x,
            (N + block.y - 1) / block.y);
  hipEvent_t start, stop;
  hipEventCreate(&start);
  hipEventCreate(&stop);
  hipEventRecord(start, 0);

  hipLaunchKernelGGL(sgemm_naive, grid, block, 0, 0,
                     M, N, K, 1.0f, d_A, d_B, 0.0f, d_C);
  e = hipGetLastError();
  printf("launch: %s\n", hipGetErrorString(e));
  if (e != hipSuccess)
    return 0;
  e = hipGetLastError();
  hipEventRecord(stop, 0);
  hipEventSynchronize(stop);

  float ms = 0.0f;

  hipEventElapsedTime(&ms, start, stop);

  hipEventDestroy(start);
  hipEventDestroy(stop);
  double flops = 2.0 * M * N * K;
  double gflops = (flops / (ms / 1000.0)) / 1e9;
  printf("Performance: %.2f GFLOPS\n", gflops);
  printf("Kernel runtime: %.3f ms\n", ms);
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
