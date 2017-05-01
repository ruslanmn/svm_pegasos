#include <iostream>
#include "mnist_data_loader.h"
#include "SVM.h"
#include "mnist_data_classifier.h"
#include <omp.h>

#include <utility>
#include <CL/cl.hpp>
#include <CL/cl.h>

using namespace std;


int main() {



    srand(time(NULL));
    cl_platform_id platform_ids[2];
    clGetPlatformIDs(2, platform_ids, NULL);

    cl_platform_id platform_id = platform_ids[0];


    // Get the GPU device associated with the platform

    cl_device_id device_ids[2];
    clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU, 2, device_ids, NULL);
    cl_device_id& device_id = device_ids[0];
    char info[1024];
    clGetDeviceInfo(device_id, CL_DEVICE_NAME, 1024, info, NULL);
    cout << info << endl;

    cl_context context = clCreateContext(NULL, 1, &device_id, NULL, NULL, NULL);

    MnistDataClassifier mdc("/home/kmeansfan/MNIST Data/images.data", "/home/kmeansfan/MNIST Data/labels.data",
                                      "/home/kmeansfan/MNIST Data/test-images.data", "/home/kmeansfan/MNIST Data/test-labels.data",
                            0.1, 1000, context, device_id);



    uint8_t* images = mdc.mdl.get_test_images();
    uint8_t* labels = mdc.mdl.get_test_labels();
    size_t test_data_size = mdc.mdl.get_test_data_size();
    size_t correct_count = 0;
    correct_count = 0;
    for(size_t i = 0; i < test_data_size; i++) {
        if( mdc.predict(&images[i * mdc.mdl.get_weight_size()]) == labels[i] )
            correct_count++;
        cout << i << endl;
    }

    cout << correct_count << "/" << test_data_size << endl;


    return 0;
}


/*

int main2() {
    cl_platform_id platform_id;
    clGetPlatformIDs(1, &platform_id, NULL);



    const char* source = "__kernel void square(__global float* input, __global float* output, int N)\n"
            "{\n"
            "    int i = get_global_id(0);\n"
            "    if ( i < N )\n"
            "       output[i] = input[i] * input[i];\n"
            "N = 0;\n"
            "}\n";

    // Get the first GPU device associated with the platform
    cl_device_id device_id;
    clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU, 2, &device_id, NULL);

    cl_context context = clCreateContext(NULL, 1, &device_id, NULL, NULL, NULL);

    cl_program program = clCreateProgramWithSource(context, 1, &source, NULL, NULL);
    if( clBuildProgram(program, 0, NULL, NULL, NULL, NULL) != CL_SUCCESS ) {
        char log[999999];
        clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, 999999, log, NULL);
        cout << log << endl;
    }

    cl_kernel kernel = clCreateKernel(program, "square", NULL);



    srand(time(NULL));



    size_t n = 99999999;
    cl_float* a = (cl_float*) malloc(sizeof(cl_float) * n);
    cl_float* b = (cl_float*) malloc(sizeof(cl_float) * n);

    for(int i = 0; i < n; i++)
        a[i] = rand() % 1000;


    cl_command_queue cmd_queue = clCreateCommandQueue(context, device_id, 0, NULL);
    cl_mem a_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(cl_float) * n, a, NULL);
    cl_mem b_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_float) * n, NULL, NULL);



    cl_int err;

    clock_t begin;

    size_t localWorkSize = 32;
    size_t numWorkGroups = (n + localWorkSize - 1) / localWorkSize;
    size_t globalWorkSize = numWorkGroups * localWorkSize;


    begin = clock();
    //err = clEnqueueWriteBuffer(cmd_queue, a_buffer, CL_FALSE, 0, n, a, NULL, NULL, NULL);
    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&a_buffer);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &b_buffer);
    clSetKernelArg(kernel, 2, sizeof(int), (void *)&n);

    err = clEnqueueNDRangeKernel(cmd_queue, kernel, 1, NULL, &globalWorkSize, &localWorkSize, 0, NULL, NULL);
    err = clFinish(cmd_queue);
    if (err != CL_SUCCESS )
        cout << "BAD" << endl;

    clEnqueueReadBuffer(cmd_queue, b_buffer, CL_TRUE, 0, sizeof(cl_float) * n, b, 0, NULL, NULL);
    cout << (clock() - begin) << endl;

    cout << "b[24] = " << b[n-1] << ", a[24] = " << a[n-1] << endl;

    begin = clock();
    for(int i = 0; i < n; i++)
        b[i] = a[i] * a[i];
    cout << (clock() - begin) << endl;
    cout << "b[24] = " << b[n-1] << ", a[24] = " << a[n-1] << endl;
}*/