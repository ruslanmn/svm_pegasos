//
// Created by kmeansfan on 4/14/17.
//

#include "mnist_data_classifier.h"

MnistDataClassifier::MnistDataClassifier(const char* train_images_filename,
                                         const char* train_labels_filename,
                                         const char* test_images_filename,
                                         const char* test_labels_filename) : mld{MnistDataLoader()} {
    mld.load_mnist_data(train_images_filename, train_labels_filename, test_images_filename, test_labels_filename);

    uint8_t** images = mld.get_train_images();
    uint8_t* labels = mld.get_train_labels();
    size_t data_size = mld.get_train_data_size();
    size_t weight_size = mld.get_weight_size();

    for(size_t i = 0; i < data_size; i++)
        number_sizes[labels[i]]++;

    for(size_t i = 0; i < 10; i++) {
        train_images_[i] = (double**) malloc(sizeof(double*) * number_sizes[i]);
        for(size_t j = 0; j < number_sizes[i]; j++)
            train_images_[i][j] = (double*)malloc(sizeof(double) * weight_size);
    }

    for(size_t i = 0; i < data_size; i++) {
        uint8_t number = labels[i];
        for( size_t image_index = 0; image_index < number_sizes[number]; image_index++)
            for( size_t pixel_index = 0; pixel_index < weight_size; pixel_index++)
                train_images_[number][image_index][pixel_index] = (double) images[i][]
    }
}