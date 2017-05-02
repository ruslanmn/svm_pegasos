//
// Created by svmfan on 3/23/17.
//

#ifndef SVM_MNIST_DATA_LOADER_H
#define SVM_MNIST_DATA_LOADER_H

#include <cstdio>
#include <stdint.h>
#include <cstdlib>


class MnistDataLoader {
private:
    uint8_t* train_images_;
    uint8_t* train_labels_;
    uint8_t* test_images_;
    uint8_t* test_labels_;
public:
    uint8_t *get_train_images() const;
    uint8_t *get_train_labels() const;
    uint8_t *get_test_images() const;
    uint8_t *get_test_labels() const;

private:
    unsigned int weight_size_;
    unsigned int train_data_size_;
    unsigned int test_data_size_;
    uint8_t* load_images(const char* images_filename, uint32_t* size);
    uint8_t* load_labels(const char* labels_filename, uint32_t* size);
public:
    MnistDataLoader();
    ~MnistDataLoader();
    unsigned int get_weight_size() const;
    unsigned int get_train_data_size() const;
    unsigned int get_test_data_size() const;
    void set_data_size(unsigned int data_size);
    void load_mnist_data(const char* train_images_filename,
                         const char* train_labels_filename,
                         const char* test_images_filename,
                         const char* test_labels_filename);
};


#endif //SVM_MNIST_DATA_LOADER_H
