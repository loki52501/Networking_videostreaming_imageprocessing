cmake_minimum_required(VERSION 3.15)
project(cpp_client_server LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(net_util src/net_util.cpp)

add_executable(server src/server.cpp)
target_link_libraries(server PRIVATE net_util)

add_executable(client src/client.cpp)
target_link_libraries(client PRIVATE net_util)

if (WIN32)
  target_link_libraries(server PRIVATE ws2_32)
  target_link_libraries(client PRIVATE ws2_32)
endif()
