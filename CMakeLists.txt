cmake_minimum_required(VERSION 3.12)
project(day1_gray_video CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)

add_executable(gray_video day1/gray_video.cpp)
target_link_libraries(gray_video PRIVATE ${OpenCV_LIBS})
target_include_directories(gray_video PRIVATE ${OpenCV_INCLUDE_DIRS})

