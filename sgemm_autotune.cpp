#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>

template <int BM, int BN, int BK, int TM, int TN>
__global__ void sgemm_kernel(int M, int N, int K,
                             float alpha,
                             const float *A,
                             const float *B,
                             float beta,
                             float *C)
{
  constexpr int NUM_THREADS = (BM / TM) * (BN / TN);

  __shared__ float As[BK * BM];
  __shared__ float Bs[BK * BN];

  const int cRow = blockIdx.y;
  const int cCol = blockIdx.x;
  const int tid = threadIdx.x;

  const int threadRow = tid / (BN / TN);
  const int threadCol = tid % (BN / TN);

  const int innerRowA = tid / (BK / 4);
  const int innerColA = tid % (BK / 4);
  const int innerRowB = tid / (BN / 4);
  const int innerColB = tid % (BN / 4);

  constexpr int strideA = NUM_THREADS / (BK / 4);
  constexpr int strideB = NUM_THREADS / (BN / 4);

  A += cRow * BM * K;
  B += cCol * BN;
  C += cRow * BM * N + cCol * BN;

  float threadResults[TM * TN] = {0.0f};
  float regM[TM] = {0.0f};
  float regN[TN] = {0.0f};

  for (int bkIdx = 0; bkIdx < K; bkIdx += BK)
  {
    for (int loadOffset = 0; loadOffset < BM; loadOffset += strideA)
    {
      float4 tmp = reinterpret_cast<const float4 *>(
          &A[(innerRowA + loadOffset) * K + innerColA * 4])[0];
      As[(innerColA * 4 + 0) * BM + innerRowA + loadOffset] = tmp.x;
      As[(innerColA * 4 + 1) * BM + innerRowA + loadOffset] = tmp.y;
      As[(innerColA * 4 + 2) * BM + innerRowA + loadOffset] = tmp.z;
      As[(innerColA * 4 + 3) * BM + innerRowA + loadOffset] = tmp.w;
    }

    for (int loadOffset = 0; loadOffset < BK; loadOffset += strideB)
    {
      reinterpret_cast<float4 *>(
          &Bs[(innerRowB + loadOffset) * BN + innerColB * 4])[0] =
          reinterpret_cast<const float4 *>(
              &B[(innerRowB + loadOffset) * N + innerColB * 4])[0];
    }

    __syncthreads();
    A += BK;
    B += BK * N;

    for (int dotIdx = 0; dotIdx < BK; dotIdx++)
    {
      for (int i = 0; i < TM; i++)
        regM[i] = As[dotIdx * BM + threadRow * TM + i];
      for (int j = 0; j < TN; j++)
        regN[j] = Bs[dotIdx * BN + threadCol * TN + j];
      for (int i = 0; i < TM; i++)
        for (int j = 0; j < TN; j++)
          threadResults[i * TN + j] += regM[i] * regN[j];
    }
    __syncthreads();
  }

  for (int i = 0; i < TM; i++)
  {
    for (int j = 0; j < TN; j += 4)
    {
      float4 tmp;
      tmp.x = alpha * threadResults[i * TN + j + 0];
      tmp.y = alpha * threadResults[i * TN + j + 1];
      tmp.z = alpha * threadResults[i * TN + j + 2];
      tmp.w = alpha * threadResults[i * TN + j + 3];
      reinterpret_cast<float4 *>(
          &C[(threadRow * TM + i) * N + threadCol * TN + j])[0] = tmp;
    }
  }
}

template <int BM, int BN, int BK, int TM, int TN>
float run_benchmark(int M, int N, int K, float *d_A, float *d_B, float *d_C)
{
  constexpr int NUM_THREADS = (BM / TM) * (BN / TN);

  dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
  dim3 block(NUM_THREADS);

  hipLaunchKernelGGL((sgemm_kernel<BM, BN, BK, TM, TN>), grid, block, 0, 0,
                     M, N, K, 1.0f, d_A, d_B, 0.0f, d_C);
  hipDeviceSynchronize();

  hipEvent_t start, stop;
  hipEventCreate(&start);
  hipEventCreate(&stop);

  hipEventRecord(start);
  for (int i = 0; i < 10; i++)
  {
    hipLaunchKernelGGL((sgemm_kernel<BM, BN, BK, TM, TN>), grid, block, 0, 0,
                       M, N, K, 1.0f, d_A, d_B, 0.0f, d_C);
  }
  hipEventRecord(stop);
  hipEventSynchronize(stop);

  float ms;
  hipEventElapsedTime(&ms, start, stop);
  hipEventDestroy(start);
  hipEventDestroy(stop);

  return ms / 10.0f;
}
#define TEST_CONFIG(BM, BN, BK, TM, TN)                                   \
  do                                                                      \
  {                                                                       \
    float ms = run_benchmark<BM, BN, BK, TM, TN>(M, N, K, d_A, d_B, d_C); \
    double gflops = (flops / (ms / 1000.0)) / 1e9;                        \
    printf("BM=%3d BN=%3d BK=%2d TM=%d TN=%d | %6.2f ms | %8.2f GFLOPS",  \
           BM, BN, BK, TM, TN, ms, gflops);                               \
    if (gflops > best_gflops)                                             \
    {                                                                     \
      best_gflops = gflops;                                               \
      printf(" << BEST");                                                 \
    }                                                                     \
    printf("\n");                                                         \
  } while (0)

int main(int argc, char **argv)
{
  if (argc < 4)
    return 1;

  int M = atoi(argv[1]);
  int N = atoi(argv[2]);
  int K = atoi(argv[3]);

  printf("--------------------------------------------\n");
  printf("SGEMM Autotuning: M=%d N=%d K=%d\n", M, N, K);
  printf("---------------------------------------------\n");

  size_t bytesA = (size_t)M * K * sizeof(float);
  size_t bytesB = (size_t)K * N * sizeof(float);
  size_t bytesC = (size_t)M * N * sizeof(float);

  float *A = (float *)malloc(bytesA);
  float *B = (float *)malloc(bytesB);
  float *C = (float *)malloc(bytesC);

  for (int i = 0; i < M * K; i++)
    A[i] = 1.0f;
  for (int i = 0; i < K * N; i++)
    B[i] = 1.0f;
  for (int i = 0; i < M * N; i++)
    C[i] = 0.0f;

  float *d_A, *d_B, *d_C;
  hipMalloc(&d_A, bytesA);
  hipMalloc(&d_B, bytesB);
  hipMalloc(&d_C, bytesC);
  hipMemcpy(d_A, A, bytesA, hipMemcpyHostToDevice);
  hipMemcpy(d_B, B, bytesB, hipMemcpyHostToDevice);
  hipMemcpy(d_C, C, bytesC, hipMemcpyHostToDevice);

  double flops = 2.0 * M * N * K;
  double best_gflops = 0.0;

  printf("\n--- 64x64 tiles ---\n");
  TEST_CONFIG(64, 64, 8, 8, 8);
  TEST_CONFIG(64, 64, 16, 8, 8);
  TEST_CONFIG(64, 64, 32, 8, 8);

  printf("\n--- 128x128 tiles ---\n");
  TEST_CONFIG(128, 128, 8, 8, 8);
  TEST_CONFIG(128, 128, 16, 8, 8);
  TEST_CONFIG(128, 128, 32, 8, 8);
  printf("\n---------------------------------\n");
  printf("Best: %.2f GFLOPS\n", best_gflops);
  printf("----------------------------------\n");

  hipMemcpy(C, d_C, bytesC, hipMemcpyDeviceToHost);
  double sum = 0.0;
  for (int i = 0; i < M * N; i++)
    sum += C[i];
  printf("Sum of all C entries: %.6f\n", sum);

  hipFree(d_A);
  hipFree(d_B);
  hipFree(d_C);
  free(A);
  free(B);
  free(C);

  return 0;
}
