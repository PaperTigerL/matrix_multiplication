cmake_minimum_required(VERSION 3.10) # 指定 CMake 版本要求
project(matrix_multiplication) # 指定项目名称

# 设置 C++ 标准版本
set(CMAKE_CXX_STANDARD 14)

# 添加头文件目录
include_directories(include)

# 添加源文件
add_executable(matrix_multiplication src/main.cpp src/matrix.cpp)

# 设置编译选项（可选）
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3") # 使用 O3 优化选项
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native") # 使用本地架构优化

# 如果需要链接第三方库，可以使用以下命令
# target_link_libraries(matrix_multiplication -l<库名>)
