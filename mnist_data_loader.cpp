#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include "mnist_data_loader.h"
#include "byteorder_helper.h"

#include <iostream>

#define SIZE 10000

using namespace std;

void MnistDataLoader::load_mnist_data(const char* train_images_filename,
		const char* train_labels_filename,
		const char* test_images_filename,
		const char* test_labels_filename
        ) {

    uint32_t size;
    // loading
    this->train_images_ = load_images(train_images_filename, &size);
    this->train_data_size_ = size;
    this->train_labels_ = load_labels(train_labels_filename, &size);
    if( train_data_size_ > size )
        this->train_data_size_ = size;

    this->test_images_ = load_images(test_images_filename, &size);
    this->test_data_size_ = size;
    this->test_labels_ = load_labels(test_labels_filename, &size);
    if( test_data_size_ > size )
        this->test_data_size_ = size;

}

uint8_t* MnistDataLoader::load_images(const char* images_filename, uint32_t* size) {
	FILE* images_file = fopen(images_filename, "rb");

	uint32_t magic;
	uint32_t rows_size;
	uint32_t columns_size;

    fread_uint32_with_flip( &magic, images_file);
    fread_uint32_with_flip( size, images_file);
    fread_uint32_with_flip( &rows_size, images_file);
    fread_uint32_with_flip( &columns_size, images_file);

    weight_size_ = (rows_size*columns_size);
    *size = SIZE;
    size_t imgs_size = *size * weight_size_;
    size_t imgs_byte_size = imgs_size * sizeof(uint8_t);

	uint8_t* images_data = (uint8_t*) malloc(imgs_byte_size);
    fread( images_data, imgs_byte_size, 1, images_file);

	fclose(images_file);

    for(size_t i = 0; i < imgs_size; i++)
            if(images_data[i] != 0)
                images_data[i] = 1;

	return images_data;
}


uint8_t* MnistDataLoader::load_labels(const char* labels_filename , uint32_t* size) {
    FILE* labels_file = fopen(labels_filename, "rb");

    uint32_t magic;

    fread_uint32_with_flip( &magic, labels_file);
    fread_uint32_with_flip( size, labels_file);
    *size = SIZE;
    uint8_t* labels_data = (uint8_t*) malloc((*size) * sizeof(uint8_t));

    fread( labels_data, sizeof(uint8_t) * (*size), 1, labels_file);

    fclose(labels_file);
    return labels_data;
}

size_t MnistDataLoader::get_weight_size() const {
    return weight_size_;
}

size_t MnistDataLoader::get_train_data_size() const {
    return train_data_size_;
}

size_t MnistDataLoader::get_test_data_size() const {
    return test_data_size_;
}

void MnistDataLoader::set_data_size(size_t data_size) {
    this->train_data_size_ = data_size;
}

MnistDataLoader::MnistDataLoader() {
    this->train_images_ = NULL;
    this->train_labels_ = NULL;
    this->test_images_ = NULL;
    this->test_labels_ = NULL;
}

MnistDataLoader::~MnistDataLoader() {
    if (train_images_ != NULL) {
        free(train_images_);
        train_images_ = NULL;
    }
    if (train_labels_ != NULL) {
        free(train_labels_);
        train_labels_ = NULL;
    }
    if (test_images_ != NULL) {
        free(test_images_);
        test_images_ = NULL;
    }
    if (test_labels_ != NULL) {
        free(test_labels_);
        test_labels_ = NULL;
    }
}

uint8_t *MnistDataLoader::get_train_images() const {
    return train_images_;
}

uint8_t *MnistDataLoader::get_train_labels() const {
    return train_labels_;
}

uint8_t *MnistDataLoader::get_test_images() const {
    return test_images_;
}

uint8_t *MnistDataLoader::get_test_labels() const {
    return test_labels_;
}
