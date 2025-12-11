#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>

#define BM 128
#define BN 128
#define BK 8
#define WM 32
#define WN 64
#define WMITER 2
#define WNITER 2
#define TM 4
#define TN 4
#define WARP_SIZE 32
#define WSUBM (WM / WMITER)
#define WSUBN (WN / WNITER)
#define NUM_WARPS_M (BM / WM)
#define NUM_WARPS_N (BN / WN)
#define NUM_WARPS (NUM_WARPS_M * NUM_WARPS_N)
#define NUM_THREADS (NUM_WARPS * WARP_SIZE)
__global__ void sgemm_warptile(int M, int N, int K,
                               float alpha,
                               const float *A,
                               const float *B,
                               float beta,
                               float *C)
{
  __shared__ float As[BK * BM];
  __shared__ float Bs[BK * BN];

  const int cRow = blockIdx.y;
  const int cCol = blockIdx.x;
  const int tid = threadIdx.x;

  const int warpId = tid / WARP_SIZE;
  const int warpRow = warpId / NUM_WARPS_N;
  const int warpCol = warpId % NUM_WARPS_N;

  const int tidInWarp = tid % WARP_SIZE;
  const int threadColInWarp = tidInWarp % (WSUBN / TN);
  const int threadRowInWarp = tidInWarp / (WSUBN / TN);

  const int innerRowA = tid / (BK / 4);
  const int innerColA = tid % (BK / 4);
  const int innerRowB = tid / (BN / 4);
  const int innerColB = tid % (BN / 4);

  constexpr int strideA = NUM_THREADS / (BK / 4);
  constexpr int strideB = NUM_THREADS / (BN / 4);

  A += cRow * BM * K;
  B += cCol * BN;
  C += cRow * BM * N + cCol * BN;

  float threadResults[WMITER * WNITER * TM * TN] = {0.0f};
  float regM[WMITER * TM] = {0.0f};
  float regN[WNITER * TN] = {0.0f};

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

      for (int wSubRowIdx = 0; wSubRowIdx < WMITER; wSubRowIdx++)
      {
        for (int i = 0; i < TM; i++)
        {
          regM[wSubRowIdx * TM + i] =
              As[dotIdx * BM +
                 warpRow * WM +
                 wSubRowIdx * WSUBM +
                 threadRowInWarp * TM + i];
        }
      }

      for (int wSubColIdx = 0; wSubColIdx < WNITER; wSubColIdx++)
      {
        for (int j = 0; j < TN; j++)
        {
          regN[wSubColIdx * TN + j] =
              Bs[dotIdx * BN +
                 warpCol * WN +
                 wSubColIdx * WSUBN +
                 threadColInWarp * TN + j];
        }
      }

      for (int wSubRowIdx = 0; wSubRowIdx < WMITER; wSubRowIdx++)
      {
        for (int wSubColIdx = 0; wSubColIdx < WNITER; wSubColIdx++)
        {
          for (int resIdxM = 0; resIdxM < TM; resIdxM++)
          {
            for (int resIdxN = 0; resIdxN < TN; resIdxN++)
            {
              int idx = (wSubRowIdx * TM + resIdxM) * (WNITER * TN) +
                        (wSubColIdx * TN) + resIdxN;
              threadResults[idx] +=
                  regM[wSubRowIdx * TM + resIdxM] *
                  regN[wSubColIdx * TN + resIdxN];
            }
          }
        }
      }
    }
    __syncthreads();
  }

  for (int wSubRowIdx = 0; wSubRowIdx < WMITER; wSubRowIdx++)
  {
    for (int wSubColIdx = 0; wSubColIdx < WNITER; wSubColIdx++)
    {

      float *C_sub = C +
                     (warpRow * WM + wSubRowIdx * WSUBM + threadRowInWarp * TM) * N +
                     (warpCol * WN + wSubColIdx * WSUBN + threadColInWarp * TN);

      for (int i = 0; i < TM; i++)
      {
        for (int j = 0; j < TN; j += 4)
        {
          int idx = (wSubRowIdx * TM + i) * (WNITER * TN) +
                    (wSubColIdx * TN) + j;
          float4 tmp;
          tmp.x = alpha * threadResults[idx + 0];
          tmp.y = alpha * threadResults[idx + 1];

          tmp.z = alpha * threadResults[idx + 2];
          tmp.w = alpha * threadResults[idx + 3];
          reinterpret_cast<float4 *>(&C_sub[i * N + j])[0] = tmp;
        }
      }
    }
  }
}

int main(int argc, char **argv)
{
  int M = 4096, N = 4096, K = 4096;
  if (argc >= 4)
  {
    M = atoi(argv[1]);
    N = atoi(argv[2]);
    K = atoi(argv[3]);
  }

  printf("Config: BM=%d BN=%d BK=%d WM=%d WN=%d TM=%d TN=%d\n",
         BM, BN, BK, WM, WN, TM, TN);
  printf("        WMITER=%d WNITER=%d NUM_THREADS=%d\n",
         WMITER, WNITER, NUM_THREADS);

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

  dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
  dim3 block(NUM_THREADS);

  printf("Grid: (%d, %d), Block: %d threads (%d warps)\n",
         grid.x, grid.y, block.x, NUM_WARPS);

  hipLaunchKernelGGL(sgemm_warptile, grid, block, 0, 0,
                     M, N, K, 1.0f, d_A, d_B, 0.0f, d_C);
  hipDeviceSynchronize();

  hipEvent_t start, stop;
  hipEventCreate(&start);
  hipEventCreate(&stop);

  hipEventRecord(start);
  for (int i = 0; i < 10; i++)
  {
    hipLaunchKernelGGL(sgemm_warptile, grid, block, 0, 0,
                       M, N, K, 1.0f, d_A, d_B, 0.0f, d_C);
  }
  hipEventRecord(stop);
  hipEventSynchronize(stop);

  float ms;
  hipEventElapsedTime(&ms, start, stop);
  float avg_ms = ms / 10.0f;
  double flops = 2.0 * M * N * K;
  double gflops = (flops / (avg_ms / 1000.0)) / 1e9;

  printf("\nKernel 10 time: %.3f ms\n", avg_ms);
  printf("Performance: %.2f GFLOPS\n", gflops);

  hipMemcpy(C, d_C, bytesC, hipMemcpyDeviceToHost);
  double sum = 0.0;
  for (int i = 0; i < M * N; i++)
    sum += C[i];
  printf("Sum of all C entries: %.6f\n", sum);

  hipEventDestroy(start);
  hipEventDestroy(stop);
  hipFree(d_A);
  hipFree(d_B);
  hipFree(d_C);
  free(A);
  free(B);
  free(C);

  return 0;
}
