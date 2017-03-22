#include <cstdio>
#include <cstdint>
#include <cstdlib>


void load_mnist_data(const char* train_images_filename,
		const char* train_labels_filename,
		const char* test_images_filename,
		const char* test_labels_filename,
		uint8_t* labels) {

	// open files

	FILE* train_labels_file  = fopen(train_labels_filename, "rb");
	FILE* test_images_file  = fopen(test_images_filename, "rb");
	FILE* test_labels_file  = fopen(test_labels_filename, "rb");





	// close files

	fclose(train_labels_file);
	fclose(test_images_file);
	fclose(test_labels_file);

}

uint8_t* load_images(const char* images_filename) {
	FILE* images_file = fopen(images_filename, "rb");

	uint32_t magic;
	uint32_t images_amount;
	uint32_t rows_amount;
	uint32_t columns_size;

	fread( &magic, 1, 4, images_file);
	fread( &images_amount, 1, 4, images_file);
	fread( &rows_amount, 1, 4, images_file);
	fread( &columns_size, 1, 4, images_file);

    uint32_t pixel_count = rows_amount*columns_size;

	uint8_t** images_data = (uint8_t**) malloc(images_amount * sizeof(uint8_t*));

	for(uint32_t image_index = 0; image_index < images_amount; image_index++) {
		images_data[image_index] = (uint8_t*) malloc(rows_amount*columns_size*sizeof(uint8_t));
        fread( images_data[image_index], sizeof(uint8_t), pixel_count,  images_file);
	}

			(uint8_t*)malloc(*rows_amount*columns_size);

	for(int pixel_count = 0; pixel_count = )

	fclose(train_images_file);
	return images_data;
}
