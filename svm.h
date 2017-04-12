//
// Created by svmfan on 3/25/17.
//

#ifndef SVM_SVM_H
#define SVM_SVM_H

#include <cstdlib>
#include <cstdint>

class svm {
    size_t data_size, weight_size;
    double* v = NULL;
    double** x = NULL;

    double (*kernel)(double*, double*, size_t);
    void free_memory();
    void set(double **x, size_t weight_size, size_t data_size, double (*kernel)(double *, double *, size_t));
public:
    svm();
    void fit(double** x, size_t weight_size, double* y, size_t data_size, double (*kernel)(double*, double*, size_t), double h, size_t T);
    double predict(double* x);
    ~svm();


};


#endif //SVM_SVM_H
