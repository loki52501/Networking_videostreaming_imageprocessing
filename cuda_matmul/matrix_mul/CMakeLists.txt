cmake_minimum_required(VERSION 3.15)
project(day5_math_memory LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(intensity_lab intensity_lab.cpp)
add_executable(tiling_sim tiling_sim.cpp)
add_executable(wave_quant wave_quant.cpp)
add_executable(tensor_core_align tensor_core_align.cpp)