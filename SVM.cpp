//
// Created by svmfan on 3/25/17.
//

#include "SVM.h"
#include <iostream>
#include <cstring>

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

SVM::SVM() {
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

void SVM::set(double** x, size_t weight_size, size_t data_size, double (*kernel)(double*, double*, size_t)) {
    this->data_size = data_size;
    this->weight_size = weight_size;
    this->kernel = kernel;

    this->x = (double**) malloc(sizeof(double*) * data_size);
    for(int i = 0; i < data_size; i++) {
        this->x[i] = (double*) malloc(sizeof(double) * weight_size);
        memcpy(this->x[i], x[i], weight_size * sizeof(double));
    }
}

void SVM::fit(double** x, size_t weight_size, double* y, size_t data_size, double (*kernel)(double*, double*, size_t), double h, size_t T) {
    free_memory();
    set(x, weight_size, data_size, kernel);

    double s;
    size_t a[data_size] = {0};
    double q;


    for(int t = 1; t <= T; t++) {
        int i = rand() % data_size;

        q = 1/(h*t);
        s = 0;
        for(int j = 0; j < data_size; j++)
            if( j != i ) {
                s += a[j] * y[j] * kernel(x[i], x[j], weight_size);
            }

        if (y[i] * q * s < 1) {
            a[i] += 1;
        }
    }

    v = (double*)malloc(sizeof(double) * data_size);
    for(int j = 0; j < data_size; j++) {
        v[j] = q * a[j] * y[j];
    }

    w = (double*) calloc(weight_size, sizeof(double));
    double cur_x[weight_size];
    for(int i = 0; i < data_size; i++) {
        memcpy(cur_x, x[i], weight_size * sizeof(double));
        produce_vector(cur_x, weight_size, v[i]);
        add_to_vector(w, cur_x, weight_size);
    }

}

double SVM::predict(double* x) {
    double res = 0;
    //for(int i = 0; i < data_size; i++)
      //  res += v[i] * this->kernel(x, this->x[i], weight_size);
    res = dot(x, w, weight_size);
    return res;
}