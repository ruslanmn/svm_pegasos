cmake_minimum_required(VERSION 3.6)
project(svm_cuda)

find_package(CUDA REQUIRED)
#include_directories( /opt/intel/intel-opencl-1.2-6.3.0.1904/opencl-sdk/include )
include_directories( /usr/local/cuda-8.0/include )

set(SOURCE_FILES main.cu mnist_data_loader.cu mnist_data_loader.h byteorder_helper.cu byteorder_helper.h SVM.cu SVM.h mnist_data_classifier.cu mnist_data_classifier.h)
cuda_add_executable(svm_cuda ${SOURCE_FILES})
#target_link_libraries(svm_cuda /opt/intel/opencl/libOpenCL.so)
#target_link_libraries(svm /usr/local/cuda/lib64/libOpenCL.so )