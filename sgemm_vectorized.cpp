#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>

#define BK 8
#define TM 8
#define TN 8
#define BM 128
#define BN 128

#define NUM_THREADS 256

__global__ void sgemm_vectorized(int M, int N, int K,
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

  const int threadRow = tid / (BN / TN);
  const int threadCol = tid % (BN / TN);

  const int innerRowA = tid / (BK / 4);
  const int innerColA = tid % (BK / 4);

  ✓ const int innerRowB = tid / (BN / 4);
  const int innerColB = tid % (BN / 4);

  A += cRow * BM * K;
  B += cCol * BN;
  C += cRow * BM * N + cCol * BN;

  float threadResults[TM * TN] = {0.0f};
  float regM[TM] = {0.0f};
  float regN[TN] = {0.0f};

  for (int bkIdx = 0; bkIdx < K; bkIdx += BK)
  {

    float4 tmp = reinterpret_cast<const float4 *>(
        &A[innerRowA * K + innerColA * 4])[0];

    As[(innerColA * 4 + 0) * BM + innerRowA] = tmp.x;
    As[(innerColA * 4 + 1) * BM + innerRowA] = tmp.y;
    As[(innerColA * 4 + 2) * BM + innerRowA] = tmp.z;
    As[(innerColA * 4 + 3) * BM + innerRowA] = tmp.w;

    reinterpret_cast<float4 *>(
        &Bs[innerRowB * BN + innerColB * 4])[0] =
        reinterpret_cast<const float4 *>(
            &B[innerRowB * N + innerColB * 4])[0];

    __syncthreads();

    A += BK;
    B += BK * N;

    for (int dotIdx = 0; dotIdx < BK; dotIdx++)
    {

      for (int i = 0; i < TM; i++)
      {
        regM[i] = As[dotIdx * BM + threadRow * TM + i];
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
    int globalRow = cRow * BM + threadRow * TM + i;
    if (globalRow < M)
    {
      for (int j = 0; j < TN; j += 4)
      {
        int globalCol = cCol * BN + threadCol * TN + j;
        if (globalCol + 3 < N)
        {
          float4 tmp;
          tmp.x = alpha * threadResults[i * TN + j + 0];
          tmp.y = alpha * threadResults[i * TN + j + 1];
          tmp.z = alpha * threadResults[i * TN + j + 2];
          tmp.w = alpha * threadResults[i * TN + j + 3];

          if (beta != 0.0f)
          {
            float4 c_val = reinterpret_cast<float4 *>(
                &C[(threadRow * TM + i) * N + threadCol * TN + j])[0];
            tmp.x += beta * c_val.x;
            tmp.y += beta * c_val.y;
            tmp.z += beta * c_val.z;
            tmp.w += beta * c_val.w;
          }

          reinterpret_cast<float4 *>(
              &C[(threadRow * TM + i) * N + threadCol * TN + j])[0] = tmp;
        }
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
    A[i] = 1.f;
  for (int i = 0; i < K * N; i++)
    B[i] = 1.f;
  for (int i = 0; i < M * N; i++)
    C[i] = 0.f;

  float *dA, *dB, *dC;
  hipMalloc(&dA, bytesA);
  hipMalloc(&dB, bytesB);
  hipMalloc(&dC, bytesC);

  hipMemcpy(dA, A, bytesA, hipMemcpyHostToDevice);
  hipMemcpy(dB, B, bytesB, hipMemcpyHostToDevice);
  hipMemcpy(dC, C, bytesC, hipMemcpyHostToDevice);
  dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
  dim3 block(NUM_THREADS);
  hipEvent_t start, stop;
  hipEventCreate(&start);
  hipEventCreate(&stop);
  hipEventRecord(start);

  hipLaunchKernelGGL(sgemm_vectorized, grid, block, 0, 0,
                     M, N, K, 1.f, dA, dB, 0.f, dC);

  hipEventRecord(stop);
  hipEventSynchronize(stop);

  float ms;
  hipEventElapsedTime(&ms, start, stop);
  printf("Vectorized 2D Tile kernel runtime: %.3f ms\n", ms);

  hipMemcpy(C, dC, bytesC, hipMemcpyDeviceToHost);

  double sum = 0.0;
  for (int i = 0; i < M * N; i++)
  {
    sum += C[i];
  }
  printf("Sum of all C entries: %.6f\n", sum);

  hipFree(dA);
  hipFree(dB);
  hipFree(dC);
  free(A);
  free(B);
  free(C);
}
