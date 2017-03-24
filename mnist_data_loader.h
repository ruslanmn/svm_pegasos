//
// Created by svmfan on 3/23/17.
//

#ifndef SVM_MNIST_DATA_LOADER_H
#define SVM_MNIST_DATA_LOADER_H

#include <cstdio>
#include <cstdint>
#include <cstdlib>

uint32_t* load_mnist_data(const char* train_images_filename,
                          const char* train_labels_filename,
                          const char* test_images_filename,
                          const char* test_labels_filename,
                          uint8_t*** ptr_train_images,
                          uint8_t** ptr_train_labels,
                          uint8_t*** ptr_test_images,
                          uint8_t** ptr_test_labels
);
uint8_t** load_images(const char* images_filename, uint32_t* size);
uint8_t* load_labels(const char* labels_filename , uint32_t* size);

#endif //SVM_MNIST_DATA_LOADER_H
