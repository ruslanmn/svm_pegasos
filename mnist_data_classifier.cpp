//
// Created by kmeansfan on 4/14/17.
//

#include <cstring>
#include "mnist_data_classifier.h"
#include "SVM.h"
#include <iostream>

using namespace std;

void convert_to_float(uint8_t* src, float* dst, size_t size) {
    for( size_t i = 0; i < size; i++)
        dst[i] = (float) src[i];
}

MnistDataClassifier::MnistDataClassifier(const char* train_images_filename,
                                         const char* train_labels_filename,
                                         const char* test_images_filename,
                                         const char* test_labels_filename,
                                         float h,
                                         size_t batch_size,
                                         cl_context context,
                                         cl_device_id device_id) {

    mdl.load_mnist_data(train_images_filename, train_labels_filename, test_images_filename, test_labels_filename);

    uint8_t* images = mdl.get_train_images();
    uint8_t* labels = mdl.get_train_labels();
    size_t data_size = mdl.get_train_data_size();
    size_t weight_size = mdl.get_weight_size();

    for(size_t i = 0; i < data_size; i++)
        number_sizes[labels[i]]++;

    for(size_t i = 0; i < 10; i++) {
        train_images_[i] = (float*) malloc(sizeof(float) * weight_size * number_sizes[i]);
    }

    size_t imagex_indices[10] = {0};
    size_t image_index;
    for(size_t i = 0; i < data_size; i++) {
        uint8_t number = labels[i];
        image_index = imagex_indices[number];
        convert_to_float(&images[i * weight_size], &train_images_[number][image_index * weight_size], weight_size);
        imagex_indices[number]++;
    }


    load_svm_classes(h, batch_size, context, device_id);
}


void MnistDataClassifier::load_svm_classes(float h, size_t batch_size, cl_context context, cl_device_id device_id) {
    int n = 10;
    svm_classes = new SVM**[n];
    size_t weight_size = mdl.get_weight_size();
    for(int i = 0; i < n; i++) {
        svm_classes[i] = new SVM*[n];
        for(int j = 0; j < n ; j++) {
            if (j <= i)
                svm_classes[i][j] = NULL;
            else {
                svm_classes[i][j] = new SVM(context, device_id);
                size_t size = number_sizes[i] + number_sizes[j];
                float* x = (float*) malloc( size * weight_size * sizeof(float) );
                memcpy(x, train_images_[i], number_sizes[i] * weight_size * sizeof(float));
                memcpy(&x[number_sizes[i] * weight_size], train_images_[j], number_sizes[j] * weight_size * sizeof(float));

                float y[size];
                for(size_t f = 0; f < number_sizes[i]; f++)
                    y[f] = 1;
                for(size_t l = number_sizes[i]; l < size; l++)
                    y[l] = -1;

                cout << "fitting " << i << "-" << j << endl;
                svm_classes[i][j]->fit(x, weight_size, y, size, h, batch_size);
                cout << "finished" << endl;

                free(x);
            }

        }
    }
}

uint8_t MnistDataClassifier::predict(uint8_t* x_uint) {
    size_t n = 10;
    size_t weight_size = mdl.get_weight_size();
    float x_float[weight_size];
    convert_to_float(x_uint, x_float, weight_size);

    size_t counts[n] = {0};
    float res;


    for(size_t i = 0; i < n; i++)
        for(size_t j = i + 1; j < n; j++) {
            res = svm_classes[i][j]->predict(x_float);
            if( res >= 1 )
                counts[i]++;
            else if (res <= -1)
                counts[j]++;
        }

    size_t most_frequent_number = 0;
    for(size_t i = 1; i < n; i++) {
        if(counts[i] > counts[most_frequent_number])
            most_frequent_number = i;
    }

    return most_frequent_number;
}

/*
int main() {
    MnistDataClassifier mdc("/home/svmfan/MNIST Data/images.data", "/home/svmfan/MNIST Data/labels.data",
                        "/home/svmfan/MNIST Data/test-images.data", "/home/svmfan/MNIST Data/test-labels.data");


    float* x[mdc.number_sizes[0] + mdc.number_sizes[1]];
    memcpy(x, mdc.train_images_[0], mdc.number_sizes[0] * sizeof(float*));
    memcpy(&x[mdc.number_sizes[0]], mdc.train_images_[1], mdc.number_sizes[1] * sizeof(float*));

    float y[mdc.number_sizes[0] + mdc.number_sizes[1]];
    for(size_t i = 0; i < mdc.number_sizes[0]; i++)
        y[i] = 1;
    for(size_t i = mdc.number_sizes[0]; i < mdc.number_sizes[0] + mdc.number_sizes[1]; i++)
        y[i] = -1;

    SVM svm;
    svm.fit(x, mdc.mdl.get_weight_size(), y, mdc.number_sizes[0] + mdc.number_sizes[1], &kernel2, 0.01, 1000);



    MnistDataLoader& mdl = mdc.mdl;
    size_t test_data_size = mdl.get_test_data_size();
    float** test_x = (float**)malloc(sizeof(float*) * test_data_size);
    for(int i = 0; i <  test_data_size; i++) {
        test_x[i] = (float *) malloc(sizeof(float) * mdl.get_weight_size());
        for(int p = 0; p < mdl.get_weight_size(); p++)
            test_x[i][p] = (float)mdl.get_test_images()[i][p];
    }


    size_t class_using = 0, class_total = 0;
    size_t win = 0;
    size_t total = 0;

    using namespace std;

    for(int i = 0; i < test_data_size; i++) {

        if (mdl.get_test_labels()[i] > 1)
            continue;

        total++;

        float c = svm.predict(test_x[i]);

        uint8_t r;
        if (c >= 0)
            r = 0;
        else
            r = 1;

        if ( mdl.get_test_labels()[i]  == r ) {
            win++;
        }

    }

    cout << win << "/" << total << endl;


    return 0;
}*/