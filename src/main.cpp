#include "matrix.h"
#include <iostream>
#include <chrono>
#include <cstdlib>
#include <ctime>
#include <random> // 引入随机数生成库
#include <functional> // 引入 std::function
#include <fstream> // 引入文件流库

// 函数声明
void testMatrixOperation(const std::function<Matrix(const Matrix&, const Matrix&)>& operation, 
                          int size, int trials, int iterations, std::ofstream& logFile);

// 封装矩阵创建、初始化和性能测试的逻辑
void testMatrixOperation(const std::function<Matrix(const Matrix&, const Matrix&)>& operation, 
                          int size, int trials, int iterations, std::ofstream& logFile) {
    double totalTime = 0.0;

    for (int iteration = 0; iteration < iterations; iteration++) {
        for (int trial = 0; trial < trials; trial++) {
            // 创建测试矩阵
            Matrix A(size, size);
            Matrix B(size, size);

            // 使用随机数生成器初始化矩阵元素
            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_int_distribution<> dis(0, 99);
            for (int i = 0; i < A.getRows(); i++) {
                for (int j = 0; j < A.getCols(); j++) {
                    A.setValue(i, j, dis(gen));
                    B.setValue(i, j, dis(gen));
                }
            }

            // 记录开始时间
            auto start = std::chrono::high_resolution_clock::now();

            // 调用传入的矩阵操作函数
            Matrix C = operation(A, B);

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

// 调用封装后的函数进行性能测试
void testMatrixMultiplication(int size, int trials, int iterations, std::ofstream& logFile) {
    // 测试普通实现
    logFile << "Testing original matrix multiplication:" << std::endl;
    testMatrixOperation(matrixMultiply, size, trials, iterations, logFile);

    // 测试优化实现
    logFile << "Testing optimized matrix multiplication:" << std::endl;
    testMatrixOperation(matrixMultiply_opt, size, trials, iterations, logFile);
}
// 测试优化后的代码计算结果是否与原始的代码一样
void testMatrixMultiplicationEquality(int size, int trials, int iterations) {
    for (int iteration = 0; iteration < iterations; iteration++) {
        for (int trial = 0; trial < trials; trial++) {
            // 创建测试矩阵
            Matrix A(size, size);
            Matrix B(size, size);

            // 使用随机数生成器初始化矩阵元素
            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_int_distribution<> dis(0, 99);
            for (int i = 0; i < A.getRows(); i++) {
                for (int j = 0; j < A.getCols(); j++) {
                    A.setValue(i, j, dis(gen));
                    B.setValue(i, j, dis(gen));
                }
            }

            // 计算原始矩阵乘法的结果
            Matrix C1 = matrixMultiply(A, B);

            // 计算优化后矩阵乘法的结果
            Matrix C2 = matrixMultiply_opt(A, B);

            // 比较两个结果矩阵
            bool isEqual = true;
            for (int i = 0; i < C1.getRows(); i++) {
                for (int j = 0; j < C1.getCols(); j++) {
                    if (std::abs(C1.getValue(i, j) - C2.getValue(i, j)) > 1e-6) {
                        isEqual = false;
                        break;
                    }
                }
                if (!isEqual) break;
            }

            // 输出比较结果
            if (trial == 0 && iteration == 0) {
                std::cout << "Original matrix multiplication result:" << std::endl;
                for (int i = 0; i < C1.getRows(); i++) {
                    for (int j = 0; j < C1.getCols(); j++) {
                        std::cout << C1.getValue(i, j) << " ";
                    }
                    std::cout << std::endl;
                }

                std::cout << "Optimized matrix multiplication result:" << std::endl;
                for (int i = 0; i < C2.getRows(); i++) {
                    for (int j = 0; j < C2.getCols(); j++) {
                        std::cout << C2.getValue(i, j) << " ";
                    }
                    std::cout << std::endl;
                }

                if (isEqual) {
                    std::cout << "The results are equal." << std::endl;
                } else {
                    std::cout << "The results are NOT equal." << std::endl;
                }
            }
        }
    }
}

int main() {
    // 设置随机数种子
    std::srand(std::time(nullptr));

    // 创建日志文件
    //std::ofstream logFile("matrix_multiplication.log");

    // 测试不同大小的矩阵
    int sizes[] = {8};
    int trials = 1; // 每个大小测试的次数
    int iterations = 1; // 每个大小测试的迭代次数

    for (int size : sizes) {
        std::cout<<"Test:"<<std::endl;
       // logFile << "Testing matrix size " << size << "x" << size << " for " << trials << " trials and " << iterations << " iterations:" << std::endl;
       // testMatrixMultiplication(size, trials, iterations, logFile);
        testMatrixMultiplicationEquality(size, trials, iterations);
    }

    //logFile.close();

    return 0;
}