cmake_minimum_required(VERSION 3.6)
project(svm)

set(CMAKE_CXX_STANDARD 14)

set(SOURCE_FILES main.cpp test.cpp)
add_executable(svm ${SOURCE_FILES})