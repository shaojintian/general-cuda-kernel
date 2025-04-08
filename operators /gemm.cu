#include <cuda_runtime.h>
#include <iostream>

#define TILE_SIZE 16

__global__ void gemm_kernel(float* A, float* B, float* C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float value = 0.0f;
        for (int i = 0; i < K; i++) {
            value += A[row * K + i] * B[i * N + col];
        }
        C[row * N + col] = value;
    }
}

extern "C" void gemm(float* A, float* B, float* C, int M, int N, int K) {
    dim3 blockDim(TILE_SIZE, TILE_SIZE);
    dim3 gridDim((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);

    gemm_kernel<<<gridDim, blockDim>>>(A, B, C, M, N, K);
    cudaDeviceSynchronize();
}