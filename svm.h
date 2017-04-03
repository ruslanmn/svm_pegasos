//
// Created by svmfan on 3/25/17.
//

#ifndef SVM_SVM_H
#define SVM_SVM_H

#include <cstdlib>
#include <cstdint>

class svm {
    size_t weight_size;
    double* w;

public:
    svm(size_t weight_size);
    void fit(uint8_t** images, uint8_t* labels, size_t data_set_size, double h, size_t T);
    double predict(uint8_t* image);
    ~svm();
};


#endif //SVM_SVM_H
