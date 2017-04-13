//
// Created by kmeansfan on 4/14/17.
//

#ifndef SVM_MNIST_DATA_CLASSIFIER_H
#define SVM_MNIST_DATA_CLASSIFIER_H


#include "mnist_data_loader.h"

class MnistDataClassifier {
private:
    MnistDataLoader mld;
    size_t number_sizes[10] = {0};
    double** train_images_[10];
public:
    
};


#endif //SVM_MNIST_DATA_CLASSIFIER_H
