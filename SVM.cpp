//
// Created by svmfan on 3/25/17.
//

#include "SVM.h"
#include <iostream>
#include <cstring>
#include <CL/cl.h>

using namespace std;

void copy(double** x, size_t weight_size, size_t data_size, double** new_x) {

}

double dot(double* x, double* b, size_t size) {
    double s = 0;
    while( size > 0 ) {
        size--;
        s += x[size] * b[size];
    }

    return s;
}

void produce_vector(double* v, size_t size, double h) {
    while( size > 0 ) {
        size--;
        v[size] *= h;
    }
}

SVM::SVM(cl_context context, cl_device_id device_id) {
    this->context = context;
    this->device_id = device_id;
}

void int_vector_to_double(uint8_t* int_x, double* new_x, size_t size) {
    while( size > 0 ) {
        size--;
        new_x[size] = int_x[size];
    }
}

void add_to_vector(double* dest, double* source, size_t size) {
    while( size > 0 ) {
        size--;
        dest[size] += source[size];
    }
}


void SVM::free_memory() {
    if (v != NULL) {
        free(v);
        v = NULL;
    }
    if (x != NULL) {
        free(x);
        v = NULL;
    }
}

SVM::~SVM() {
    free_memory();
}

void SVM::set(cl_double* x, uint weight_size, uint data_size) {
    this->data_size = data_size;
    this->weight_size = weight_size;
    this->x = x;
}


int SVM::fit(cl_double* x, cl_uint weight_size, cl_double* y, cl_uint data_size, cl_double h, cl_uint T) {
    free_memory();
    set(x, weight_size, data_size);

    cl_uint a[data_size] = {0};

    const char* src = "double kernel_func(__global double* x, __global double* b, size_t size) {\n"
            "    double s = 0;\n"
            "    while( size > 0 ) {\n"
            "        size--;\n"
            "        s += x[size] * b[size];\n"
            "    }\n"
            "\n"
            "    return s;\n"
            "}\n"
            "\n"
            "\n"
            "__kernel void fit(__global uint* indices, uint T, __global uint* a, __global double* x, uint weight_size, __global double* y, uint data_size, double h) {\n"
            "    uint id = get_global_id(0);\n"
            "\n"
            "    if (id >= T)\n"
            "        return;\n"
            "\n"
            "    uint t =  id + 1;\n"
            "    uint i = indices[id];\n"
            "\n"
            "    double q = 1/(h*t);\n"
            "    double s = 0;\n"
            "\n"
            "    for(int j = 0; j < data_size; j++)\n"
            "     if( j != i ) {\n"
            "         s += a[j] * y[j] * kernel_func(&x[i * weight_size], &x[j * weight_size], weight_size);\n"
            "     }\n"
            "\n"
            "    if (y[i] * q * s < 1) {\n"
            "     atomic_add(&a[i], 1);\n"
            "    }\n"
            "}\n";

    cl_command_queue cmd_queue = clCreateCommandQueueWithProperties(context, device_id, 0, NULL);

    cl_program program = clCreateProgramWithSource(context, 1, &src, NULL, NULL);

    if( clBuildProgram(program, 0, NULL, NULL, NULL, NULL) != CL_SUCCESS ) {
        char log[999999];
        clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, 999999, log, NULL);
        cerr << log << endl;
        std::exit(-1);
        return -1;
    }

    cl_kernel opencl_kernel = clCreateKernel(program, "fit", NULL);

    cl_mem a_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(cl_uint) * data_size, a, NULL);
    cl_mem x_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(cl_double) * weight_size * data_size, x, NULL);
    cl_mem y_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(cl_double) * data_size, x, NULL);

    cl_uint indices[T];
    for(int t = 0; t < T; t++)
        indices[t] = rand() % data_size;

    cl_mem indices_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(cl_uint) * T, indices, NULL);

    clSetKernelArg(opencl_kernel, 0, sizeof(cl_mem), &indices_buffer);
    clSetKernelArg(opencl_kernel, 1, sizeof(cl_uint), &T);
    clSetKernelArg(opencl_kernel, 2, sizeof(cl_mem), &a_buffer);
    clSetKernelArg(opencl_kernel, 3, sizeof(cl_mem), &x_buffer);
    clSetKernelArg(opencl_kernel, 4, sizeof(cl_uint), &weight_size);
    clSetKernelArg(opencl_kernel, 5, sizeof(cl_mem), &y_buffer);
    clSetKernelArg(opencl_kernel, 6, sizeof(cl_uint), &data_size);
    clSetKernelArg(opencl_kernel, 7, sizeof(cl_double), &h);

    size_t localWorkSize = 256;
    size_t numWorkGroups = (T + localWorkSize - 1) / localWorkSize;
    size_t globalWorkSize = numWorkGroups * localWorkSize;

    clock_t begin = clock();

    clEnqueueNDRangeKernel(cmd_queue, opencl_kernel, 1, NULL, &globalWorkSize, &localWorkSize, 0, NULL, NULL);
    clFinish(cmd_queue);
    cout << (clock() - begin) << endl;

    clEnqueueReadBuffer(cmd_queue, a_buffer, CL_TRUE, 0, sizeof(cl_uint) * data_size, a, 0, NULL, NULL);

    clReleaseMemObject(indices_buffer);
    clReleaseMemObject(a_buffer);
    clReleaseMemObject(x_buffer);
    clReleaseMemObject(y_buffer);


    double q = 1/(h*T);
    v = (double*)malloc(sizeof(double) * data_size);
    for(int j = 0; j < data_size; j++) {
        v[j] = q * a[j] * y[j];
    }

    w = (double*) calloc(weight_size, sizeof(double));
    double cur_x[weight_size];
    for(int i = 0; i < data_size; i++) {
        memcpy(cur_x, &x[i * weight_size], weight_size * sizeof(double));
        produce_vector(cur_x, weight_size, v[i]);
        add_to_vector(w, cur_x, weight_size);
    }

    return 0;
}

double SVM::predict(double* x) {
    double res = 0;
    //for(int i = 0; i < data_size; i++)
      //  res += v[i] * this->kernel(x, this->x[i], weight_size);
    res = dot(x, w, weight_size);
    return res;
}