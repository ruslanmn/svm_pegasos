//
// Created by kmeansfan on 4/14/17.
//

#ifndef SVM_MNIST_DATA_CLASSIFIER_H
#define SVM_MNIST_DATA_CLASSIFIER_H


#include "mnist_data_loader.h"
#include "SVM.h"

class MnistDataClassifier {
private:
    size_t number_sizes[10] = {0};
    double* train_images_[10];
    SVM*** svm_classes = NULL;
public:
    MnistDataLoader mdl;
    MnistDataClassifier(const char* train_images_filename,
                         const char* train_labels_filename,
                         const char* test_images_filename,
                         const char* test_labels_filename,
                         double h,
                        size_t batch_size,
                        cl_context context,
                        cl_device_id device_id);
    void load_svm_classes(double h, size_t batch_size, cl_context context, cl_device_id device_id);
    uint8_t predict(uint8_t* x);
};


#endif //SVM_MNIST_DATA_CLASSIFIER_H
