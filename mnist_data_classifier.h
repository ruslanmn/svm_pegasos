//
// Created by kmeansfan on 4/14/17.
//

#ifndef SVM_MNIST_DATA_CLASSIFIER_H
#define SVM_MNIST_DATA_CLASSIFIER_H


#include "mnist_data_loader.h"
#include "SVM.h"

class MnistDataClassifier {
private:
    unsigned int number_sizes[10];
    float* train_images_[10];
    SVM*** svm_classes;
public:
    MnistDataLoader mdl;
    MnistDataClassifier(const char* train_images_filename,
                         const char* train_labels_filename,
                         const char* test_images_filename,
                         const char* test_labels_filename,
                         float h,
                        unsigned int batch_size);
    void load_svm_classes(float h, unsigned int batch_size);
    uint8_t predict(uint8_t* x);
};


#endif //SVM_MNIST_DATA_CLASSIFIER_H
