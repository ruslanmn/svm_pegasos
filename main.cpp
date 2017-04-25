#include <iostream>
#include "mnist_data_loader.h"
#include "SVM.h"
#include "mnist_data_classifier.h"
#include <omp.h>


using namespace std;

void mnist_to_double(uint8_t** images, uint8_t* labels, double** x, double* y, size_t weight_size, size_t data_size) {
    for(int i = 0; i < data_size; i++)
        for(int j = 0; j < weight_size; j++) {
            x[i][j] = (double) images[i][j];
        }

    for(int i = 0; i < data_size; i++)
        if( labels[i] == 0 )
            y[i] = -1;
        else
            y[i] = 1;
}

double kernel(double* x, double* b, size_t size) {
    double s = 0;
    while( size > 0 ) {
        size--;
        s += x[size] * b[size];
    }

    return s;
}


int main() {
    srand(time(NULL));

    MnistDataClassifier mdc("/home/svmfan/MNIST Data/images.data", "/home/svmfan/MNIST Data/labels.data",
                                      "/home/svmfan/MNIST Data/test-images.data", "/home/svmfan/MNIST Data/test-labels.data",
                            0.1, 300, kernel);



    uint8_t** images = mdc.mdl.get_test_images();
    uint8_t* labels = mdc.mdl.get_test_labels();
    size_t test_data_size = mdc.mdl.get_test_data_size();
    size_t correct_count = 0;
    correct_count = 0;
#pragma omp parallel for
    for(size_t i = 0; i < test_data_size; i++) {
        if( mdc.predict(images[i]) == labels[i] )
            correct_count++;
        cout << i << endl;
    }

    cout << correct_count << "/" << test_data_size << endl;


    return 0;
}
