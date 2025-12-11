#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>

#define BK 8
#define TM 8
#define TN 8
#define BM 64
#define BN 64
#define NUM_THREADS 64

__global__ void sgemm_2dtile(int M, int N, int K,
                             float alpha,
                             const float *A,
                             const float *B,
                             float beta,
                             float *C)
{
  __shared__ float As[BM * BK];
  __shared__ float Bs[BK * BN];

  int tid = threadIdx.x;

  int threadRow = tid / (BN / TN);
  int threadCol = tid % (BN / TN);

  int strideA = NUM_THREADS / BK;
  int innerRowA = tid / BK;
  int innerColA = tid % BK;

  int strideB = NUM_THREADS / BN;
  int innerRowB = tid / BN;
  int innerColB = tid % BN;

  int blockRow = blockIdx.y * BM;
  int blockCol = blockIdx.x * BN;

  const float *Ap = A + blockRow * K;
  const float *Bp = B + blockCol;
  float *Cp = C + blockRow * N + blockCol;

  float threadResults[TM * TN] = {0.0f};
  float regM[TM], regN[TN];

  for (int bkIdx = 0; bkIdx < K; bkIdx += BK)
  {

    for (int loadOffset = 0; loadOffset < BM; loadOffset += strideA)
    {
      int aRow = innerRowA + loadOffset;
      if ((blockRow + aRow) < M && (bkIdx + innerColA) < K)
        As[aRow * BK + innerColA] = Ap[aRow * K + innerColA];
      else
        As[aRow * BK + innerColA] = 0.0f;
    }

    for (int loadOffset = 0; loadOffset < BK; loadOffset += strideB)
    {
      int bRow = innerRowB + loadOffset;
      if ((bkIdx + bRow) < K && (blockCol + innerColB) < N)
        Bs[bRow * BN + innerColB] = Bp[bRow * N + innerColB];
      else
        Bs[bRow * BN + innerColB] = 0.0f;
    }

    __syncthreads();

    Ap += BK;
    Bp += BK * N;

    for (int dotIdx = 0; dotIdx < BK; dotIdx++)
    {

      for (int i = 0; i < TM; i++)
      {
        regM[i] = As[(threadRow * TM + i) * BK + dotIdx];
      }

      for (int j = 0; j < TN; j++)
      {
        regN[j] = Bs[dotIdx * BN + threadCol * TN + j];
      }

      for (int i = 0; i < TM; i++)
      {
        for (int j = 0; j < TN; j++)
        {
          threadResults[i * TN + j] += regM[i] * regN[j];
        }
      }
    }

    __syncthreads();
  }

  for (int i = 0; i < TM; i++)
  {
    for (int j = 0; j < TN; j++)
    {
      int globalRow = blockRow + threadRow * TM + i;
      int globalCol = blockCol + threadCol * TN + j;
      if (globalRow < M && globalCol < N)
      {
        Cp[(threadRow * TM + i) * N + (threadCol * TN + j)] =
            alpha * threadResults[i * TN + j] +
            beta * Cp[(threadRow * TM + i) * N + (threadCol * TN + j)];
      }
    }
  }
}
int main(int argc, char **argv)
{
  if (argc < 4)
    return 1;

  int M = atoi(argv[1]);
  int N = atoi(argv[2]);
  int K = atoi(argv[3]);

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
  hipEvent_t start, stop;
  hipEventCreate(&start);
  hipEventCreate(&stop);
  hipEventRecord(start);

  hipLaunchKernelGGL(sgemm_2dtile, grid, block, 0, 0,
                     M, N, K, 1.0f, d_A, d_B, 0.0f, d_C);

  hipEventRecord(stop);
  hipEventSynchronize(stop);

  float ms;
  hipEventElapsedTime(&ms, start, stop);
  printf("2D Tile kernel runtime: %.3f ms\n", ms);

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
