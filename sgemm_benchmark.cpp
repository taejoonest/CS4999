#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>
#include <cstdio>
#include <cstdlib>

#define IDX_CM(row, col, ld) ((col) * (ld) + (row))

int main(int argc, char **argv)
{
  int M = 4096, N = 4096, K = 4096;
  if (argc >= 4)
  {
    M = atoi(argv[1]);
    N = atoi(argv[2]);
    K = atoi(argv[3]);
  }

  size_t size;
  rocblas_get_version_string_size(&size);
  char *version = (char *)malloc(size);
  rocblas_get_version_string(version, size);
  printf("rocBLAS version: %s\n\n", version);
  free(version);

  size_t bytesA = (size_t)M * K * sizeof(float);
  size_t bytesB = (size_t)K * N * sizeof(float);
  size_t bytesC = (size_t)M * N * sizeof(float);

  printf("Memory: A=%.1fMB, B=%.1fMB, C=%.1fMB\n",
         bytesA / 1e6, bytesB / 1e6, bytesC / 1e6);

  float *A = (float *)malloc(bytesA);
  float *B = (float *)malloc(bytesB);
  float *C = (float *)malloc(bytesC);

  for (int col = 0; col < K; col++)
  {
    for (int row = 0; row < M; row++)
    {
      A[IDX_CM(row, col, M)] = 1.0f;
    }
  }
  for (int col = 0; col < N; col++)
  {
    for (int row = 0; row < K; row++)
    {
      B[IDX_CM(row, col, K)] = 1.0f;
    }
  }
  for (int col = 0; col < N; col++)
  {
    for (int row = 0; row < M; row++)
    {
      C[IDX_CM(row, col, M)] = 0.0f;
    }
  }

  float *d_A, *d_B, *d_C;
  hipMalloc(&d_A, bytesA);
  hipMalloc(&d_B, bytesB);
  hipMalloc(&d_C, bytesC);
  hipMemcpy(d_A, A, bytesA, hipMemcpyHostToDevice);
  hipMemcpy(d_B, B, bytesB, hipMemcpyHostToDevice);
  hipMemcpy(d_C, C, bytesC, hipMemcpyHostToDevice);

  float alpha = 1.0f, beta = 0.0f;
  double flops = 2.0 * M * N * K;

  rocblas_handle handle;
  rocblas_status status = rocblas_create_handle(&handle);
  if (status != rocblas_status_success)
  {
    printf("ERROR: rocblas_create_handle failed with status %d\n", status);
    return 1;
  }

  hipEvent_t start, stop;
  hipEventCreate(&start);
  hipEventCreate(&stop);

  printf("\nRunning rocblas_sgemm...\n");
  printf("  transA=N, transB=N\n");
  printf("  M=%d, N=%d, K=%d\n", M, N, K);
  printf("  lda=%d, ldb=%d, ldc=%d\n", M, K, M);

  for (int i = 0; i < 5; i++)
  {
    status = rocblas_sgemm(handle,
                           rocblas_operation_none,
                           rocblas_operation_none,
                           M, N, K,
                           &alpha,
                           d_A, M,
                           d_B, K,
                           &beta,
                           d_C, M);
    if (status != rocblas_status_success)
    {
      printf("ERROR: rocblas_sgemm failed with status %d\n", status);
      return 1;
    }
  }
  hipDeviceSynchronize();

  // Benchmark
  hipEventRecord(start);
  for (int i = 0; i < 10; i++)
  {
    rocblas_sgemm(handle,
                  rocblas_operation_none,
                  rocblas_operation_none,
                  M, N, K,
                  &alpha,
                  d_A, M,
                  d_B, K,
                  &beta,
                  d_C, M);
    if (status != rocblas_status_success)
    {
      printf("ERROR: rocblas_sgemm failed with status %d\n", status);
      return 1;
    }
  }
  hipDeviceSynchronize();

  hipEventRecord(start);
  for (int i = 0; i < 10; i++)
  {
    rocblas_sgemm(handle,
                  rocblas_operation_none,
                  rocblas_operation_none,
                  M, N, K,
                  &alpha,
                  d_A, M,
                  d_B, K,
                  &beta,
                  d_C, M);
  }
  hipEventRecord(stop);
  hipEventSynchronize(stop);

  float ms;
  hipEventElapsedTime(&ms, start, stop);
  float avg_ms = ms / 10.0f;
  double gflops = (flops / (avg_ms / 1000.0)) / 1e9;

  printf("\n--- Results ---\n");
  printf("Time: %.3f ms\n", avg_ms);
  printf("Performance: %.2f GFLOPS\n", gflops);

  hipMemcpy(C, d_C, bytesC, hipMemcpyDeviceToHost);
  double sum = 0.0;
  for (int i = 0; i < M * N; i++)
    sum += C[i];
  printf("Sum: %.0f (expect %.0f)\n", sum, (double)M * N * K);

  printf("C[0,0] = %.1f (expect %d)\n", C[IDX_CM(0, 0, M)], K);
  printf("C[M-1,N-1] = %.1f (expect %d)\n", C[IDX_CM(M - 1, N - 1, M)], K);

  if (sum != (double)M * N * K)
  {
    printf("WARNING: Sum mismatch! Result may be incorrect.\n");
  }

  hipEventDestroy(start);
  hipEventDestroy(stop);
  rocblas_destroy_handle(handle);
  hipFree(d_A);
  hipFree(d_B);
  hipFree(d_C);
  free(A);
  free(B);
  free(C);

  return 0;
}
