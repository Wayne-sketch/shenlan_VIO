cmake_minimum_required(VERSION 2.8)
project(Q2)
set(CMAKE_CXX_FLAGS "-std=c++17")
# 为使用 sophus，需要使用find_package命令找到它
find_package(Sophus REQUIRED)

include_directories("/usr/include/eigen3")

ADD_EXECUTABLE(Q2 Q2.cpp)
TARGET_LINK_LIBRARIES(Q2 Sophus::Sophus)