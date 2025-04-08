#include <cuda_runtime.h>
#include <torch/torch.h>
#include <iostream>
#include <chrono>

// 引入 CUDA 函数
extern "C" void gemm(float* A, float* B, float* C, int M, int N, int K);

int main() {
    // 矩阵维度
    int M = 1024; // 行数
    int N = 1024; // 列数
    int K = 1024; // 中间维度

    // 创建 PyTorch 张量
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    torch::Tensor A = torch::rand({M, K}, options);
    torch::Tensor B = torch::rand({K, N}, options);
    torch::Tensor C = torch::zeros({M, N}, options);

    // 获取原始指针
    float* A_ptr = A.data_ptr<float>();
    float* B_ptr = B.data_ptr<float>();
    float* C_ptr = C.data_ptr<float>();

    // 测试自定义 GEMM
    auto start = std::chrono::high_resolution_clock::now();
    gemm(A_ptr, B_ptr, C_ptr, M, N, K);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float, std::milli> duration = end - start;
    std::cout << "Custom GEMM Time: " << duration.count() << " ms" << std::endl;

    // 测试 PyTorch 官方 GEMM
    start = std::chrono::high_resolution_clock::now();
    torch::Tensor C_ref = torch::matmul(A, B);
    end = std::chrono::high_resolution_clock::now();
    duration = end - start;
    std::cout << "Torch matmul Time: " << duration.count() << " ms" << std::endl;

    // 验证结果
    bool is_close = torch::allclose(C, C_ref, 1e-5, 1e-5);
    std::cout << "Results match: " << (is_close ? "Yes" : "No") << std::endl;

    return 0;
}