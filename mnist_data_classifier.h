//
// Created by kmeansfan on 4/14/17.
//

#ifndef SVM_MNIST_DATA_CLASSIFIER_H
#define SVM_MNIST_DATA_CLASSIFIER_H


#include "mnist_data_loader.h"

class MnistDataClassifier {
private:
    MnistDataLoader& mld;
    size_t number_sizes[10] = {0};
    double** train_images_[10];
public:
    MnistDataClassifier(const char* train_images_filename,
                         const char* train_labels_filename,
                         const char* test_images_filename,
                         const char* test_labels_filename);
};


#endif //SVM_MNIST_DATA_CLASSIFIER_H
