#ifndef MATRIX_H
#define MATRIX_H

#include <vector>
#include <stdexcept>
#include <iostream>

// 多维矩阵类
class MultidimensionalMatrix {
public:
    MultidimensionalMatrix(const std::vector<int>& dims) : dims(dims) {
        // 初始化矩阵数据
        data.resize(1); // 初始化一个一维数据向量
        for (int i = 0; i < dims.size(); i++) {
            data.resize(data.size() * dims[i]); // 扩展数据向量以匹配当前维度的大小
        }
    }

    // 设置矩阵元素
    void setValue(const std::vector<int>& indices, double value) {
        if (indices.size() != dims.size()) {
            throw std::out_of_range("Index out of range");
        }
        int index = 0;
        for (int i = 0; i < indices.size(); i++) {
            index = index * dims[i] + indices[i];
        }
        data[index] = value;
    }

    // 获取矩阵元素
    double getValue(const std::vector<int>& indices) const {
        if (indices.size() != dims.size()) {
            throw std::out_of_range("Index out of range");
        }
        int index = 0;
        for (int i = 0; i < indices.size(); i++) {
            index = index * dims[i] + indices[i];
        }
        return data[index];
    }

    // 获取矩阵维度
    const std::vector<int>& getDims() const { return dims; }

private:
    std::vector<int> dims; // 存储矩阵维度
    std::vector<double> data; // 存储矩阵数据
};
// 矩阵乘法函数
// 普通实现
MultidimensionalMatrix matrixMultiply(const MultidimensionalMatrix& A, const MultidimensionalMatrix& B);
// 优化实现
MultidimensionalMatrix matrixMultiply_opt(const MultidimensionalMatrix& A, const MultidimensionalMatrix& B);
#endif // MATRIX_H
