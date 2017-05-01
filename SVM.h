//
// Created by svmfan on 3/25/17.
//

#ifndef SVM_SVM_H
#define SVM_SVM_H

#include <cstdlib>
#include <cstdint>
#include <CL/cl.h>

class SVM {
    cl_uint data_size, weight_size;
    float* v = NULL;
    cl_float* x = NULL;
    float* w;

    cl_context context;
    cl_device_id device_id;

    void free_memory();
    void set(cl_float *x, uint weight_size, uint data_size);
public:
    SVM(cl_context context, cl_device_id device_id);
    int fit(cl_float* x, cl_uint weight_size, cl_float* y, cl_uint data_size, cl_float h, cl_uint T);
    float predict(float* x);
    ~SVM();


};


#endif //SVM_SVM_H
