cmake_minimum_required(VERSION 3.10)
project(FER_TensorRT_Deploy)
# 开启 Release 模式和 O3 终极优化，让 C++ 代码全速运行！
set(CMAKE_BUILD_TYPE Release)

# 设置 C++ 标准
set(CMAKE_CXX_STANDARD 14)

# 1. 寻找 OpenCV 和 CUDA 库
find_package(OpenCV REQUIRED)
find_package(CUDA REQUIRED)

# 2. 配置头文件路径
include_directories(${CUDA_INCLUDE_DIRS})
include_directories(/usr/include/aarch64-linux-gnu)

# 3. 配置库文件搜索路径 (这里是关键！新增了 CUDA 的路径)
link_directories(/usr/local/cuda/lib64)
link_directories(/usr/lib/aarch64-linux-gnu)

# 4. 生成可执行文件
add_executable(fer_app main.cpp)

# 5. 链接库
target_include_directories(fer_app PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(fer_app 
    ${OpenCV_LIBS} 
    nvinfer        
    nvonnxparser   
    cudart         
)
