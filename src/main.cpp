#include "matrix.h"
#include <iostream>
#include <chrono>
#include <cstdlib>
#include <ctime>
#include <random>
#include <functional>
#include <fstream>

// 函数声明
void testMatrixOperation(const std::function<MultidimensionalMatrix(const MultidimensionalMatrix&, const MultidimensionalMatrix&)>& operation,
                         const std::vector<int>& dims, int trials, int iterations, std::ofstream& logFile);

// 封装矩阵创建、初始化和性能测试的逻辑
void testMatrixOperation(const std::function<MultidimensionalMatrix(const MultidimensionalMatrix&, const MultidimensionalMatrix&)>& operation,
                         const std::vector<int>& dims, int trials, int iterations, std::ofstream& logFile) {
    double totalTime = 0.0;

    for (int iteration = 0; iteration < iterations; iteration++) {
        for (int trial = 0; trial < trials; trial++) {
            // 创建测试矩阵
            MultidimensionalMatrix A(dims);
            MultidimensionalMatrix B(dims);

            // 使用随机数生成器初始化矩阵元素
            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_int_distribution<> dis(0, 99);
            std::vector<int> indices(dims.size(), 0);
            std::function<void(const std::vector<int>&, std::vector<int>&)> initializeMatrix = [&](const std::vector<int>& currentDims, std::vector<int>& currentIndices) {
                if (currentDims.size() == 1) {
                    for (int i = 0; i < currentDims[0]; i++) {
                        A.setValue(currentIndices, dis(gen));
                        B.setValue(currentIndices, dis(gen));
                        currentIndices.back()++;
                    }
                } else {
                    for (int i = 0; i < currentDims[0]; i++) {
                        currentIndices[currentDims.size() - 1] = i;
                        initializeMatrix(std::vector<int>(currentDims.begin() + 1, currentDims.end()), currentIndices);
                    }
                }
            };
            initializeMatrix(dims, indices);

            // 记录开始时间
            auto start = std::chrono::high_resolution_clock::now();

            // 调用传入的矩阵操作函数
            MultidimensionalMatrix C = operation(A, B);

            // 记录结束时间
            auto end = std::chrono::high_resolution_clock::now();

            // 计算执行时间
            std::chrono::duration<double> elapsed = end - start;
            totalTime += elapsed.count();
        }
    }

    // 输出平均执行时间
    double averageTime = totalTime / (trials * iterations);
    logFile << "Average execution time: " << averageTime << " seconds" << std::endl;
}

int main() {
    // 设置随机数种子
    std::srand(std::time(nullptr));

    // 创建日志文件
    std::ofstream logFile("matrix_multiplication.log");

    // 测试不同大小的矩阵
    std::vector<int> sizes = {4,8,16,32}; // 选择 4 的倍数作为维度大小
    int trials = 1;
    int iterations = 1; // 每个大小测试的迭代次数

    for (int size : sizes) {
        // 创建测试矩阵的维度
        std::vector<int> dims = {size, size, size, size}; 
        
        // 测试矩阵尺寸
        logFile << "Testing matrix size ";
        for (int dim : dims) {
            logFile << dim << "x";
        }
        logFile << " for " << trials << " trials and " << iterations << " iterations:" << std::endl;

        // 测试普通实现
        logFile << "Testing original matrix multiplication:" << std::endl;
        testMatrixOperation(matrixMultiply, dims, trials, iterations, logFile);

        // 测试优化实现
        logFile << "Testing optimized matrix multiplication:" << std::endl;
        testMatrixOperation(matrixMultiply_opt, dims, trials, iterations, logFile);
    }

    logFile.close();

    return 0;
}
