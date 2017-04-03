//
// Created by svmfan on 3/25/17.
//

#include "svm.h"
#include <cstring>
#include <cstdio>

double dot(double* x, double* w, size_t size) {
    double s = 0;
    while( size > 0 ) {
        size--;
        s += x[size] * w[size];
    }

    return s;
}

void produce_vector(double* v, size_t size, double h) {
    while( size > 0 ) {
        size--;
        v[size] *= h;
    }
}

svm::svm(size_t weight_size) {
    this->weight_size = weight_size;
    w = (double*)malloc(sizeof(double) * weight_size);
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


svm::~svm() {
    delete[] w;
}

void svm::fit(uint8_t** images, uint8_t* labels, size_t data_set_size, double h, size_t T) {
    memset(w, 0, weight_size * sizeof(double));
    double s;
    double x[weight_size];
    for(int t = 1; t <= T; t++) {
        int i = rand() % data_set_size;
        uint8_t* int_x = images[i];
        int_vector_to_double(int_x, x, weight_size);
        uint8_t y = labels[i];
        double q = 1/(h*t);
        s = dot(x, w, weight_size);
        produce_vector(w, weight_size, 1-q*h);
        if (y * s < 1) {
            produce_vector(x, weight_size, q * y);
            add_to_vector(w, x, weight_size);
        }
    }
}

double svm::predict(uint8_t* image) {
    double x[weight_size];
    int_vector_to_double(image, x, weight_size);
    return dot(x, w, weight_size);
}