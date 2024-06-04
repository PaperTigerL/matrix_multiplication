#include "matrix.h"
#include <omp.h>
#include <arm_neon.h>


Matrix matrixMultiply_opt(const Matrix& A, const Matrix& B) {
    if (A.getCols() != B.getRows()) {
        throw std::invalid_argument("Matrix dimensions are not compatible for multiplication.");
    }

    Matrix result(A.getRows(), B.getCols());
    int blockSize = 4; // 定义循环块大小，根据实际情况进行调整
    int numThreads = omp_get_max_threads(); // 获取最大线程数

    // 分配线程工作负载
    int rowsPerThread = A.getRows() / numThreads;
    int remainingRows = A.getRows() % numThreads;

    #pragma omp parallel for schedule(dynamic) num_threads(numThreads) 
    for (int threadId = 0; threadId < numThreads; threadId++) {
        int startRow = threadId * rowsPerThread + (threadId < remainingRows ? threadId : remainingRows);
        int endRow = startRow + rowsPerThread;
        if (threadId == numThreads - 1) {
            endRow += remainingRows; // 处理剩余行
        }

        for (int j = 0; j < B.getCols(); j++) {
            float32x4_t sum = vdupq_n_f32(0.0f);
            float tempA[4], tempB[4];
            for (int k = 0; k < A.getCols(); k += blockSize) {
                // 循环展开
                for (int i = 0; i < blockSize; i++) {
                    if (k + i < A.getCols()) {
                        tempA[i] = static_cast<float>(A.getValue(startRow, k + i));
                        tempB[i] = static_cast<float>(B.getValue(k + i, j));
                    } else {
                        tempA[i] = 0.0f;
                        tempB[i] = 0.0f;
                    }
                }
                // NEON 乘加
                float32x4_t a0 = vld1q_f32(tempA);
                float32x4_t b0 = vld1q_f32(tempB);
                sum = vaddq_f32(sum, vmulq_f32(a0, b0));
            }
            // 存储结果
            double resultValue = vgetq_lane_f32(sum, 0) + vgetq_lane_f32(sum, 1) + vgetq_lane_f32(sum, 2) + vgetq_lane_f32(sum, 3);
            result.setValue(startRow, j, resultValue);
        }
    }
    return result;
}

// 普通实现的矩阵乘法
Matrix matrixMultiply(const Matrix& A, const Matrix& B) {
    // 检查矩阵维度是否匹配
    if (A.getCols() != B.getRows()) {
        throw std::invalid_argument("Matrix dimensions do not match for multiplication");
    }
    Matrix result(A.getRows(), B.getCols());
    // 进行矩阵乘法运算
    for (int i = 0; i < A.getRows(); i++) {
        for (int j = 0; j < B.getCols(); j++) {
            double sum = 0.0;
            for (int k = 0; k < A.getCols(); k++) {
                sum += A.getValue(i, k) * B.getValue(k, j);
            }
            result.setValue(i, j, sum);
        }
    }
    return result;
}