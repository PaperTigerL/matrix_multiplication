#ifndef MATRIX_H
#define MATRIX_H

#include <vector>
#include <stdexcept>

// 矩阵类
class Matrix {
public:
    Matrix(int rows, int cols) : data(rows, std::vector<double>(cols, 0.0)), rows(rows), cols(cols) {}

    // 设置矩阵元素
    void setValue(int row, int col, double value) {
        if (row >= 0 && row < rows && col >= 0 && col < cols) {
            data[row][col] = value;
        } else {
            throw std::out_of_range("Index out of range");
        }
    }

    // 获取矩阵元素
    double getValue(int row, int col) const {
        if (row >= 0 && row < rows && col >= 0 && col < cols) {
            return data[row][col];
        } else {
            throw std::out_of_range("Index out of range");
        }
    }

    // 获取矩阵行数
    int getRows() const { return rows; }

    // 获取矩阵列数
    int getCols() const { return cols; }

private:
    std::vector<std::vector<double>> data; // 存储矩阵数据
    int rows; // 行数
    int cols; // 列数
};

// 矩阵乘法函数
//普通实现
Matrix matrixMultiply(const Matrix& A, const Matrix& B);
//优化实现
Matrix matrixMultiply_opt(const Matrix&A,const Matrix& B);

#endif // MATRIX_H
