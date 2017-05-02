//
// Created by svmfan on 3/25/17.
//

#ifndef SVM_SVM_H
#define SVM_SVM_H

#include <cstdlib>
#include <stdint.h>

class SVM {
    unsigned int data_size, weight_size;
    float* v;
    float* x;
    float* w;


    void free_memory();
    void set(float *x, unsigned int weight_size, unsigned int data_size);
public:
    SVM();
    int fit(float* x, unsigned int weight_size, float* y, unsigned int data_size, float h, unsigned int T);
    float predict(float* x);
    ~SVM();


};


#endif //SVM_SVM_H
