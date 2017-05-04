//
// Created by svmfan on 3/25/17.
//

#include "SVM.h"
#include <iostream>
#include <cstring>
#include <cstdlib>
#include <cstdio>

using namespace std;


void checkError() {
    cudaError err = cudaGetLastError();
    if ( cudaSuccess != err) {
        cout << err<< endl;
        exit(-1);
    }
}

#include <cmath>


__host__ __device__ float kernel_func(float* x, float* b, unsigned int size) {
    float s = 0;
    while( size > 0 ) {
        size--;
        s += (x[size] - b[size])*(x[size] - b[size]);
    }

    return exp(-0.02 * s);
}

__global__ void fit_kernel(unsigned int id_offset,
                           unsigned int* indices, unsigned int T, unsigned int* a, float* x,
                      unsigned int weight_size, float* y, unsigned int data_size, float h, float* kernel_values) {
    unsigned int id = blockIdx.x + id_offset;
    if (id >= T)
        return;

    unsigned int t =  id + 1;
    unsigned int i = indices[id];

    float q = 1/(h*t);
    float s = 0;

    for(unsigned int j = 0; j < data_size; j++)
        if( j != i ) {
            s += a[j] * y[j] * kernel_values[id * data_size + j];
        }

    if (y[i] * q * s < 1) {
        atomicAdd(&a[i], 1);
    }
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

unsigned int* SVM::d_a = NULL;
float* SVM::d_x = NULL;
float* SVM::d_y = NULL;

void SVM::loadCuda(unsigned int max_data_size, unsigned int weight_size) {
    unsigned int a_size = sizeof(unsigned int) * max_data_size;
    cudaMalloc(&d_a, a_size);

    unsigned int x_size = sizeof(float) * weight_size * max_data_size;
    cudaMalloc(&d_x, x_size);

    unsigned int y_size = sizeof(float) * max_data_size;
    cudaMalloc(&d_y, y_size);
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

__global__ void kernel_values_compute(unsigned int t_offset, unsigned int data_size_offset,
                                      unsigned int* indices, float* kernel_values, float* x,
                                      unsigned int weight_size,
                                      unsigned int data_size, unsigned int T) {
    unsigned int t = blockIdx.x + t_offset;
    indices[t]++;
    if( t >= T )
        return;
    unsigned int j = threadIdx.x + data_size_offset;
    if( j >= data_size )
        return;

    unsigned int i = indices[t];
    kernel_values[t * data_size + j] = kernel_func(&x[i * weight_size], &x[j * weight_size], weight_size);
}

float* load_kernel_values(unsigned int* d_indices, unsigned int T, float* d_x, unsigned int weight_size, unsigned int data_size) {
    size_t size = sizeof(float) * T * data_size;
    float* d_kernel_values;
    cudaMalloc(&d_kernel_values, size);
    unsigned int max_threads = 1024;
    unsigned int max_blocks = 65535;
    unsigned int t_offset = 0;
    // t_offset = how much t we already performed
    while( T > t_offset ) {
        // data_size_offset = how much data we already performed for the current t
        unsigned int data_size_offset = 0;
        while (data_size > data_size_offset) {
            kernel_values_compute <<< max_blocks, max_threads >>> (t_offset, data_size_offset, d_indices, d_kernel_values, d_x, weight_size, data_size, T);
            data_size_offset += max_threads;
        }
        t_offset += max_blocks;
    }



    return d_kernel_values;

}

int SVM::fit(float* x, unsigned int weight_size, float* y, unsigned int data_size, float h, unsigned int T) {
    free_memory();
    //data_size = data_size % 100;

    set(x, weight_size, data_size);


    unsigned int* a;
    cudaMallocHost(&a, data_size*sizeof(unsigned int));

    unsigned int indices[T];
    for(unsigned int t = 0; t < T; t++)
        indices[t] = rand() % data_size;

    size_t total_size = 0;

    // copying to device memory
    unsigned int* d_indices;
    unsigned int indices_size = sizeof(unsigned int) * T;
    total_size += indices_size;
    cudaMalloc(&d_indices, indices_size);
    cudaMemcpy(d_indices, indices, indices_size, cudaMemcpyHostToDevice);

    unsigned int a_size = sizeof(unsigned int) * data_size;
    cudaMemset(d_a, 0, a_size);

    unsigned int x_size = sizeof(float) * weight_size * data_size;
    cudaMemcpy(d_x, x, x_size, cudaMemcpyHostToDevice);

    unsigned int y_size = sizeof(float) * data_size;
    cudaMemcpy(d_y, y, y_size, cudaMemcpyHostToDevice);


    checkError();
    float* d_kernel_values = load_kernel_values(d_indices, T, d_x, weight_size, data_size);
    //cout << total_size << endl;
    cudaThreadSynchronize();
    //cout << "Starting kernel " << endl;

    unsigned int max_threads = 1024;
    unsigned int max_blocks = 65535;
    unsigned int t_offset = 0;
    // t_offset = how much t we already performed
    while( T > t_offset ) {
        // data_size_offset = how much data we already performed for the current t
        checkError();
        fit_kernel<<<max_blocks, 1>>>(t_offset, d_indices, T, d_a, d_x, weight_size, d_y, data_size, h, d_kernel_values);
        t_offset += max_blocks;
    }

    cudaThreadSynchronize();
    //cout << "Finish kernel " << endl;
    cout << endl;
    for(int i = 0; i < data_size; i++)
        cout << a[i];
    cudaMemcpy(a, d_a, a_size, cudaMemcpyDeviceToHost);
    checkError();
    cout << endl;

    for(int i = 0; i < data_size; i++)
        cout << a[i];
    cout.flush();
    float q = 1/(h*T);
    v = (float*)malloc(sizeof(float) * data_size);
    for(unsigned int j = 0; j < data_size; j++) {
        v[j] = q * a[j] * y[j];
    }

    cudaFree(d_indices);
    cudaFree(d_kernel_values);
    cudaFreeHost(a);
    checkError();
    return 0;
}



float SVM::predict(float* x) {
    float res = 0;
    for(int i = 0; i < data_size; i++)
        res += v[i] * kernel_func(x, &(this->x[i * weight_size]), weight_size);
    return res;
}

void SVM::save(const char* filename) {
    FILE* f = fopen(filename, "wb");
    fwrite(&(this->data_size), sizeof(unsigned int), 1, f);
    fwrite(v, sizeof(float) * this->data_size, 1, f);
    fwrite(&(this->weight_size), sizeof(unsigned int), 1, f);
    fwrite(x, sizeof(float) * this->data_size * this->weight_size, 1, f);
    fclose(f);
}

bool SVM::load(const char* filename) {
    FILE* f = NULL;
    if (f = fopen(filename, "rb")) {
        free_memory();

        fread(&(this->data_size), sizeof(unsigned int), 1, f);

        size_t v_size = sizeof(float) * this->data_size;
        v = (float*)malloc(v_size);
        fread(v, v_size, 1, f);

        fread(&(this->weight_size), sizeof(unsigned int), 1, f);

        size_t x_size = v_size * weight_size;
        x = (float*)malloc(x_size);
        fread(x, x_size, 1, f);

        fclose(f);
        return true;
    }
    return false;
}