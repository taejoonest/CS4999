#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#define BK 8
#define TM 8
#define BM 64
#define BN 64
#define NUM_THREADS ((BM * BN) / TM)

__global__ void sgemm_1dtile(int M, int N, int K, float alpha,
                             const float *A, const float *B,
                             float beta, float *C)
{
  __shared__ float As[BM * BK];
  __shared__ float Bs[BK * BN];

  int tid = threadIdx.x;

  int threadRow = tid / BN;
  int threadCol = tid % BN;

  int blockRow = blockIdx.y * BM;
  int blockCol = blockIdx.x * BN;

  const float *Ap = A + blockRow * K;
  const float *Bp = B + blockCol;
  float *Cp = C + blockRow * N + blockCol;

  float threadResults[TM] = {0.0f};

  int innerRowA = tid / BK;
  int innerColA = tid % BK;
  int innerRowB = tid / BN;
  int innerColB = tid % BN;

  for (int bkIdx = 0; bkIdx < K; bkIdx += BK)
  {

    if ((blockRow + innerRowA) < M && (bkIdx + innerColA) < K)
      As[innerRowA * BK + innerColA] = Ap[innerRowA * K + innerColA];
    else
      As[innerRowA * BK + innerColA] = 0.0f;

    if ((bkIdx + innerRowB) < K && (blockCol + innerColB) < N)
      Bs[innerRowB * BN + innerColB] = Bp[innerRowB * N + innerColB];
    else
      Bs[innerRowB * BN + innerColB] = 0.0f;

    __syncthreads();

    Ap += BK;
    Bp += BK * N;

    for (int dotIdx = 0; dotIdx < BK; dotIdx++)
    {
      float Btmp = Bs[dotIdx * BN + threadCol];
      for (int resIdx = 0; resIdx < TM; resIdx++)
      {
        int aRow = threadRow * TM + resIdx;
        threadResults[resIdx] += As[aRow * BK + dotIdx] * Btmp;
      }
    }
    __syncthreads();
  }

  for (int resIdx = 0; resIdx < TM; resIdx++)
  {
    int globalRow = blockRow + threadRow * TM + resIdx;
    int globalCol = blockCol + threadCol;
    if (globalRow < M && globalCol < N)
    {
      Cp[(threadRow * TM + resIdx) * N + threadCol] =
          alpha * threadResults[resIdx] +
          beta * Cp[(threadRow * TM + resIdx) * N + threadCol];
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

  dim3 grid(int(N / BN), int(M / BM));
  dim3 block(32 * 32);

  hipEvent_t start, stop;
  hipEventCreate(&start);
  hipEventCreate(&stop);
  hipEventRecord(start);

  hipLaunchKernelGGL(sgemm_1dtile, grid, block, 0, 0,
                     M, N, K, 1.0f, d_A, d_B, 0.0f, d_C);

  hipEventRecord(stop);
  hipEventSynchronize(stop);

  float ms;
  hipEventElapsedTime(&ms, start, stop);
  printf("1D Tile kernel runtime: %.3f ms\n", ms);

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
