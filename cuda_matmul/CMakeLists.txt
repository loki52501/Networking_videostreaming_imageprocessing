cmake_minimum_required(VERSION 3.18)
project(cuda_matmul LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(CUDAToolkit REQUIRED)

add_executable(matmul src/main.cu)
set_target_properties(matmul PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "75;86")

target_compile_definitions(matmul PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:_USE_MATH_DEFINES>)
target_link_libraries(matmul PRIVATE CUDA::cudart CUDA::cublas)
