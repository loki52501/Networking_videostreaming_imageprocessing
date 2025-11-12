cmake_minimum_required(VERSION 3.16)
project(day5_thread_labs CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_executable(day5_condition_variable 1_condition_variable_basics.cpp)
add_executable(day5_safe_queue 2_safe_queue_demo.cpp)
add_executable(day5_thread_pool 3_thread_pool_stub.cpp)
