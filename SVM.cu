//
// Created by svmfan on 3/25/17.
//

#include "SVM.h"
#include <iostream>
#include <cstring>

using namespace std;




/*__device__ float kernel_func(float* x, float* b, unsigned int size) {
    float s = 0;
    while( size > 0 ) {
        size--;
        s += x[size] * b[size];
    }

    return s;
}*/

__global__ void fit_kernel() {/*unsigned int* indices, unsigned int T, unsigned int* a, float* x,
                      unsigned int weight_size, float* y, unsigned int data_size, float h) {
    /*cuPrintf("here I am");
    unsigned int id = threadIdx.x;
    unsigned int t =  id + 1;
    unsigned int i = indices[id];

    float q = 1/(h*t);
    float s = 0;

    for(unsigned int j = 0; j < data_size; j++)
        if( j != i ) {
            double k_result = 0;
            for( int i = 0; i < weight_size; i++)
                k_result += x[i * weight_size + i] * x[j * weight_size + i];
            s += a[j] * y[j] * k_result;
        }

    if (y[i] * q * s < 1) {
        atomicAdd(&a[i], 1);
    }*/
}

float dot(float* x, float* b, unsigned int size) {
    float s = 0;
    while( size > 0 ) {
        size--;
        s += x[size] * b[size];
    }

    return s;
}

void produce_vector(float* v, unsigned int size, float h) {
    while( size > 0 ) {
        size--;
        v[size] *= h;
    }
}

SVM::SVM() {
    v = NULL;
    w = NULL;
    x = NULL;
}

void int_vector_to_float(uint8_t* int_x, float* new_x, unsigned int size) {
    while( size > 0 ) {
        size--;
        new_x[size] = int_x[size];
    }
}


void add_to_vector(float* dest, float* source, unsigned int size) {
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

void SVM::set(float* x, unsigned int weight_size, unsigned int data_size) {
    this->data_size = data_size;
    this->weight_size = weight_size;
    this->x = x;
}




int SVM::fit(float* x, unsigned int weight_size, float* y, unsigned int data_size, float h, unsigned int T) {
    free_memory();

    set(x, weight_size, data_size);


    unsigned int* a = (unsigned int*)calloc(data_size, sizeof(unsigned int));
    unsigned int indices[T];
    for(unsigned int t = 0; t < T; t++)
        indices[t] = rand() % data_size;


    // copying to device memory
    unsigned int* d_indices;
    unsigned int indices_size = sizeof(unsigned int) * T;
    cudaMalloc(&d_indices, indices_size);
    cudaMemcpy(d_indices, indices, indices_size, cudaMemcpyHostToDevice);

    unsigned int* d_a;
    unsigned int a_size = sizeof(unsigned int) * data_size;
    cudaMalloc(&d_a, a_size);
    cudaMemcpy(d_a, a, a_size, cudaMemcpyHostToDevice);

    float* d_x;
    unsigned int x_size = sizeof(float) * weight_size * data_size;
    cudaMalloc(&d_x, x_size);
    cudaMemcpy(d_x, x, x_size, cudaMemcpyHostToDevice);

    float* d_y;
    unsigned int y_size = sizeof(float) * data_size;
    cudaMalloc(&d_y, y_size);
    cudaMemcpy(d_y, y, y_size, cudaMemcpyHostToDevice);
    cout << "Starting kernel " << endl;
    cudaError err = cudaGetLastError();
    if ( cudaSuccess != err) {
        cout << err<< endl;
        exit(-1);
    }
    fit_kernel<<<1, 1>>>();//d_indices, T, d_a, d_x, weight_size, d_y, data_size, h);

    cout << "Finish kernel " << endl;
    cudaMemcpy(a, d_a, a_size, cudaMemcpyDeviceToHost);

    float q = 1/(h*T);
    v = (float*)malloc(sizeof(float) * data_size);
    for(unsigned int j = 0; j < data_size; j++) {
        v[j] = q * a[j] * y[j];
    }

    w = (float*) calloc(weight_size, sizeof(float));
    float cur_x[weight_size];
    for(unsigned int i = 0; i < data_size; i++) {
        memcpy(cur_x, &x[i * weight_size], weight_size * sizeof(float));
        produce_vector(cur_x, weight_size, v[i]);
        add_to_vector(w, cur_x, weight_size);
    }
    cudaFree(d_a);
    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_indices);
    free(a);
    return 0;
}



float SVM::predict(float* x) {
    float res = 0;
    //for(int i = 0; i < data_size; i++)
      //  res += v[i] * this->kernel(x, this->x[i], weight_size);
    res = dot(x, w, weight_size);
    return res;
}