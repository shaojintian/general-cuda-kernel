#include <cuda_runtime.h>
#include <torch/torch.h>
#include <iostream>
#include <chrono>

// 引入自定义 CUDA MaxPool2D 函数
extern "C" void maxpool2d(float* input, float* output, int H, int W, int PH, int PW);

int main() {
    // 输入张量和池化窗口的维度
    int H = 1024; // 输入高度
    int W = 1024; // 输入宽度
    int PH = 2;   // 池化窗口高度
    int PW = 2;   // 池化窗口宽度

    // 创建 PyTorch 张量
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    torch::Tensor input = torch::rand({1, 1, H, W}, options); // 单通道输入
    torch::Tensor output = torch::zeros({1, 1, H / PH, W / PW}, options); // 输出张量

    // 获取原始指针
    float* input_ptr = input.data_ptr<float>();
    float* output_ptr = output.data_ptr<float>();

    // 测试自定义 CUDA MaxPool2D
    auto start = std::chrono::high_resolution_clock::now();
    maxpool2d(input_ptr, output_ptr, H, W, PH, PW);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float, std::milli> duration = end - start;
    std::cout << "Custom CUDA MaxPool2D Time: " << duration.count() << " ms" << std::endl;

    // 测试 PyTorch 官方 MaxPool2D
    start = std::chrono::high_resolution_clock::now();
    torch::Tensor output_ref = torch::nn::functional::max_pool2d(input, {PH, PW});
    end = std::chrono::high_resolution_clock::now();
    duration = end - start;
    std::cout << "Torch MaxPool2D Time: " << duration.count() << " ms" << std::endl;

    // 验证结果
    bool is_close = torch::allclose(output, output_ref, 1e-5, 1e-5);
    std::cout << "Results match: " << (is_close ? "Yes" : "No") << std::endl;

    return 0;
}