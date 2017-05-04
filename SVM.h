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
    static float* d_x;
    static float* d_y;
    static unsigned int* d_a;
    static void loadCuda(unsigned int max_data_size, unsigned int weight_size);
    SVM();
    int fit(float* x, unsigned int weight_size, float* y, unsigned int data_size, float h, unsigned int T);
    float predict(float* x);
    void save(const char* filename);
    bool load(const char* filename);
    ~SVM();


};


#endif //SVM_SVM_H
