cmake_minimum_required(VERSION 3.6)
project(svm)

set(CMAKE_CXX_STANDARD 14)

set(SOURCE_FILES main.cpp mnist_data_loader.cpp mnist_data_loader.h byteorder_helper.cpp byteorder_helper.h SVM.cpp SVM.h mnist_data_classifier.cpp mnist_data_classifier.h)
add_executable(svm ${SOURCE_FILES})