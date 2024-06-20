#include <iostream>
#include "matrix.h"
#include <vector>
#include <thread>
#include <arm_neon.h> // 仅在支持NEON的ARM架构上使用
#include <cmath>
MultidimensionalMatrix matrixMultiply_opt(const MultidimensionalMatrix& A, const MultidimensionalMatrix& B) {
    // 确保A和B的维度是兼容的
    if (A.getDims().size() != B.getDims().size() || A.getDims().back() != B.getDims()[0]) {
        throw std::invalid_argument("Matrices are incompatible for multiplication.");
    }

    // 计算结果矩阵的维度
    std::vector<int> resultDims(A.getDims().size());
    std::copy(A.getDims().begin(), A.getDims().end() - 1, resultDims.begin()); // 复制A的前n-1个维度
    std::copy(B.getDims().begin() + 1, B.getDims().end(), resultDims.end() - 1); // 复制B的后n-1个维度

    MultidimensionalMatrix result(resultDims);
    // 获取硬件支持的线程数量
    const int numThreads = std::thread::hardware_concurrency();

    // 创建线程向量
    std::vector<std::thread> threads;

    // 计算每个线程需要处理的迭代次数
    const int iterationsPerThread = ceil(resultDims.back() / static_cast<double>(numThreads));

    // 启动线程
    for (int i = 0; i < numThreads; ++i) {
        threads.emplace_back([&, i]() {
            std::vector<int> commonDimIndices(resultDims.size(), 0);
            for (int j = i * iterationsPerThread; j < resultDims.back(); j += numThreads) {
                commonDimIndices.back() = j;
                float sum[4] = {0.0f, 0.0f, 0.0f, 0.0f}; // 使用数组存储四个累加值
                  float32x4_t vecSum = vdupq_n_f32(0.0f); // 使用零向量初始化
                for (int k = 0; k < A.getDims().back(); ++k) {
                    // 创建与A维度相同的idxForA
                    std::vector<int> idxForA(A.getDims().size(), 0);
                    std::copy(commonDimIndices.begin(), commonDimIndices.end() - 1, idxForA.begin());
                    idxForA.back() = k; // 将 k 放在 A 的最后一个维度上

                    // 创建与B维度相同的idxForB
                    std::vector<int> idxForB(B.getDims().size(), 0);
                    idxForB.front() = k; // 将 k 放在 B 的第一个维度上
                    std::copy(commonDimIndices.begin() + 1, commonDimIndices.end(), idxForB.begin() + 1); // 将 commonDimIndices 的内容复制到 idxForB 的第二个维度开始的位置

               // 使用 NEON 指令进行向量乘法和累加
            float32x4_t vecA = vdupq_n_f32(A.getValue(idxForA));
            float32x4_t vecB = vdupq_n_f32(B.getValue(idxForB));
             vecSum = vfmaq_f32(vecSum, vecA, vecB); // 向量乘法和累加
                }
                // 将累加结果设置到结果矩阵中
              // 将累加结果合并为一个 double 值
double sumValue = vgetq_lane_f32(vecSum, 0) + vgetq_lane_f32(vecSum, 1) + vgetq_lane_f32(vecSum, 2) + vgetq_lane_f32(vecSum, 3);

// 将累加结果设置到结果矩阵中
result.setValue(commonDimIndices, sumValue);
            }
        });
    }

    // 等待所有线程完成
    for (auto& thread : threads) {
        thread.join();
    }

    return result;
}

MultidimensionalMatrix matrixMultiply(const MultidimensionalMatrix& A, const MultidimensionalMatrix& B) {
    // 确保A和B的维度是兼容的
    if (A.getDims().size() != B.getDims().size() || A.getDims().back() != B.getDims()[0]) {
        throw std::invalid_argument("Matrices are incompatible for multiplication.");
    }

    // 计算结果矩阵的维度
    std::vector<int> resultDims(A.getDims().size());
    std::copy(A.getDims().begin(), A.getDims().end() - 1, resultDims.begin()); // 复制A的前n-1个维度
    std::copy(B.getDims().begin() + 1, B.getDims().end(), resultDims.end() - 1); // 复制B的后n-1个维度

    MultidimensionalMatrix result(resultDims);

    // 实现多维矩阵乘法
    std::vector<int> commonDimIndices(resultDims.size(), 0); // 初始化索引向量，大小与结果矩阵的维度相同

    // 外循环遍历结果矩阵的所有维度
    while (true) {
        double sum = 0.0;
        // 计算当前索引下的乘积项并累加到sum
        for (int k = 0; k < A.getDims().back(); ++k) {
            // 创建与A维度相同的idxForA
            std::vector<int> idxForA(A.getDims().size(), 0);
            std::copy(commonDimIndices.begin(), commonDimIndices.end() - 1, idxForA.begin());
            idxForA.back() = k; // 将 k 放在 A 的最后一个维度上

            // 创建与B维度相同的idxForB
            std::vector<int> idxForB(B.getDims().size(), 0);
            idxForB.front() = k; // 将 k 放在 B 的第一个维度上
            std::copy(commonDimIndices.begin() + 1, commonDimIndices.end(), idxForB.begin() + 1); // 将 commonDimIndices 的内容复制到 idxForB 的第二个维度开始的位置

            sum += A.getValue(idxForA) * B.getValue(idxForB);
        }
        // 将乘积项设置到结果矩阵中
        result.setValue(commonDimIndices, sum);

        // 更新索引向量，准备下一个位置的计算
        int nextDim = resultDims.size() - 1; // 从最后一个维度开始递增
        while (nextDim >= 0 && ++commonDimIndices[nextDim] == resultDims[nextDim]) {
            commonDimIndices[nextDim] = 0; // 重置当前维度，前进到更高维度
            --nextDim;
            if (nextDim < 0) break; // 所有维度都已遍历完，跳出循环
        }
        if (nextDim < 0) break; // 所有维度都已遍历完，跳出循环
    }


    return result;
}
