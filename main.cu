#include <iostream>
#include "mnist_data_loader.h"
#include "SVM.h"
#include "mnist_data_classifier.h"
#include <omp.h>


using namespace std;


int print_train_images() {
    MnistDataLoader mld;
    mld.load_mnist_data("/home/kmeansfan/MNIST Data/images.data", "/home/kmeansfan/MNIST Data/labels.data",
                        "/home/kmeansfan/MNIST Data/test-images.data", "/home/kmeansfan/MNIST Data/test-labels.data");
    uint8_t* labels = mld.get_train_labels();
    uint8_t* images = mld.get_train_images();
    int a;
    for(int i = 0; i < mld.get_train_data_size(); i++) {
        for(int y = 0; y < 28; y++) {
            for(int x = 0; x < 28; x++) {
                if (images[i * mld.get_weight_size() + y*28 + x] == 0)
                    cout << " ";
                else
                    cout << "*";
            }
            cout << endl;
        }
        cout << (int)labels[i] << endl;
        cin >> a;

    }
    return 0;
}

int main() {

    int nDevices;

    cudaGetDeviceCount(&nDevices);
    for (int i = 0; i < 1; i++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        printf("Device Number: %d\n", i);
        printf("  Device name: %s\n", prop.name);
        printf("  Memory Clock Rate (KHz): %d\n",
               prop.memoryClockRate);
        printf("  Memory Bus Width (bits): %d\n",
               prop.memoryBusWidth);
        printf("  Peak Memory Bandwidth (GB/s): %f\n\n",
               2.0*prop.memoryClockRate*(prop.memoryBusWidth/8)/1.0e6);
    }


    srand(time(NULL));

    MnistDataClassifier mdc("/home/kmeansfan/MNIST Data/images.data", "/home/kmeansfan/MNIST Data/labels.data",
                                      "/home/kmeansfan/MNIST Data/test-images.data", "/home/kmeansfan/MNIST Data/test-labels.data",
                            0.1, 10);



    uint8_t* images = mdc.mdl.get_test_images();
    uint8_t* labels = mdc.mdl.get_test_labels();
    unsigned int test_data_size = mdc.mdl.get_test_data_size();
    unsigned int correct_count = 0;
    correct_count = 0;
    for(unsigned int i = 0; i < test_data_size; i++) {
        if( mdc.predict(&images[i * mdc.mdl.get_weight_size()]) == labels[i] )
            correct_count++;
        cout << i << endl;
    }

    cout << correct_count << "/" << test_data_size << endl;


    return 0;
}
/*
int main_opencl_test() {
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



    unsigned int n = 99999999;
    cl_float* a = (cl_float*) malloc(sizeof(cl_float) * n);
    cl_float* b = (cl_float*) malloc(sizeof(cl_float) * n);

    for(int i = 0; i < n; i++)
        a[i] = rand() % 1000;


    cl_command_queue cmd_queue = clCreateCommandQueue(context, device_id, 0, NULL);
    cl_mem a_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(cl_float) * n, a, NULL);
    cl_mem b_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_float) * n, NULL, NULL);



    cl_int err;

    clock_t begin;

    unsigned int localWorkSize = 32;
    unsigned int numWorkGroups = (n + localWorkSize - 1) / localWorkSize;
    unsigned int globalWorkSize = numWorkGroups * localWorkSize;


    begin = clock();
    //err = clEnqueueWriteBuffer(cmd_queue, a_buffer, CL_FALSE, 0, n, a, NULL, NULL, NULL);
    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &a_buffer);
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
  /*  for(int i = 0; i < n; i++)
        b[i] = a[i] * a[i];
    cout << (clock() - begin) << endl;
    cout << "b[24] = " << b[n-1] << ", a[24] = " << a[n-1] << endl;
}*/