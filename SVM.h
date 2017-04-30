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
    double* v = NULL;
    cl_double* x = NULL;
    double* w;

    cl_context context;
    cl_device_id device_id;

    void free_memory();
    void set(cl_double *x, uint weight_size, uint data_size);
public:
    SVM(cl_context context, cl_device_id device_id);
    int fit(cl_double* x, cl_uint weight_size, cl_double* y, cl_uint data_size, cl_double h, cl_uint T);
    double predict(double* x);
    ~SVM();


};


#endif //SVM_SVM_H
