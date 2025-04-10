#include <cuda_runtime.h>
#include <iostream>

__global__ void matrix_add(float* A, float* B, float* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        int idx = row * N + col;
        C[idx] = A[idx] + B[idx];
    }
}

int main() {
    int N = 1024;
    size_t size = N * N * sizeof(float);
    // 测试 CUDA 官方 API
    {
        float *d_A, *d_B, *d_C;
        auto start = std::chrono::high_resolution_clock::now();

        cudaMalloc(&d_A, size);
        cudaMalloc(&d_B, size);
        cudaMalloc(&d_C, size);

        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<float, std::milli> duration = end - start;
        std::cout << "CUDA API Allocation/Deallocation Time: " << duration.count() << " ms" << std::endl;
    }

    // 测试自定义内存池
    {
        MemoryPool mempool(size * 3); // 创建一个足够大的内存池
        auto start = std::chrono::high_resolution_clock::now();

        float* d_A = static_cast<float*>(mempool.allocate(size));
        float* d_B = static_cast<float*>(mempool.allocate(size));
        float* d_C = static_cast<float*>(mempool.allocate(size));

        mempool.reset(); // 重置内存池

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<float, std::milli> duration = end - start;
        std::cout << "MemoryPool Allocation/Deallocation Time: " << duration.count() << " ms" << std::endl;
    }

    return 0;
}