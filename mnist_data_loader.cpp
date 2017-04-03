#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include "mnist_data_loader.h"
#include "byteorder_helper.h"

uint32_t* load_mnist_data(const char* train_images_filename,
		const char* train_labels_filename,
		const char* test_images_filename,
		const char* test_labels_filename,
        uint8_t*** ptr_train_images,
        uint8_t** ptr_train_labels,
        uint8_t*** ptr_test_images,
        uint8_t** ptr_test_labels
        ) {

    uint32_t* sizes = (uint32_t*)malloc(4 * sizeof(uint32_t));

    // loading
    *ptr_train_images = load_images(train_images_filename, &sizes[0]);
    *ptr_train_labels = load_labels(train_labels_filename, &sizes[1]);
    *ptr_test_images = load_images(test_images_filename, &sizes[2]);
    *ptr_test_labels = load_labels(test_labels_filename, &sizes[3]);

    return sizes;
}

uint8_t** load_images(const char* images_filename, uint32_t* size) {
	FILE* images_file = fopen(images_filename, "rb");

	uint32_t magic;
	uint32_t rows_size;
	uint32_t columns_size;

    fread_uint32_with_flip( &magic, images_file);
    fread_uint32_with_flip( size, images_file);
    fread_uint32_with_flip( &rows_size, images_file);
    fread_uint32_with_flip( &columns_size, images_file);

    uint32_t pixel_count = rows_size*columns_size;

	uint8_t** images_data = (uint8_t**) malloc(*size * sizeof(uint8_t*));

	for(uint32_t image_index = 0; image_index < *size; image_index++) {
		images_data[image_index] = (uint8_t*) malloc(rows_size*columns_size*sizeof(uint8_t));
        fread( images_data[image_index], sizeof(uint8_t) * pixel_count, 1, images_file);
	}


	fclose(images_file);
	return images_data;
}

uint8_t* load_labels(const char* labels_filename , uint32_t* size) {
    FILE* labels_file = fopen(labels_filename, "rb");

    uint32_t magic;

    fread_uint32_with_flip( &magic, labels_file);
    fread_uint32_with_flip( size, labels_file);

    uint8_t* labels_data = (uint8_t*) malloc((*size) * sizeof(uint8_t));

    fread( labels_data, sizeof(uint8_t) * (*size), 1, labels_file);
    for(int i = 0; i < *size; i++) {
        if ( labels_data[i] == 4 )
            labels_data[i] == 1;
        else
            labels_data[i] == -1;
    }
    fclose(labels_file);
    return labels_data;
}