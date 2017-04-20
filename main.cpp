#include <iostream>
#include "mnist_data_loader.h"
#include "SVM.h"


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


int main2() {
    srand(time(NULL));

    MnistDataLoader mdl;
    mdl.load_mnist_data("/home/kmeansfan/MNIST Data/images.data", "/home/kmeansfan/MNIST Data/labels.data",
                                      "/home/kmeansfan/MNIST Data/test-images.data", "/home/kmeansfan/MNIST Data/test-labels.data");


    int win = 0;

    size_t weight_size = mdl.get_weight_size();

    SVM s;

    int n = 1000;

    double** train_x = (double**)malloc(sizeof(double*) * n);
    for(int i = 0; i < n; i++)
        train_x[i] = (double*)malloc(sizeof(double) * weight_size);
    double train_y[n];

    mnist_to_double(mdl.get_train_images(), mdl.get_train_labels(), train_x, train_y, weight_size, n);
    cout << "start fitting" << endl;
    s.fit(train_x, weight_size, train_y, n, &kernel, 0.01, 1000);

    size_t test_data_size = mdl.get_test_data_size();
    double** test_x = (double**)malloc(sizeof(double*) * test_data_size);
    for(int i = 0; i <  test_data_size; i++)
        test_x[i] = (double*)malloc(sizeof(double) * weight_size);
    double test_y[test_data_size];

    mnist_to_double(mdl.get_test_images(), mdl.get_test_labels(), test_x, test_y, weight_size, test_data_size);


    size_t class_using = 0, class_total = 0;
    test_data_size = 1000;
    for(int i = 0; i < test_data_size; i++) {
        double c = s.predict(test_x[i]);

        if( test_y[i] >= 0 )
            class_total++;

        int r;
        if (c >= 0) {
            r = 1;
        }
        else
            r = -1;

        if ( ( (test_y[i] > 0) && (r > 0) ) ||
                ( (test_y[i] < 0) && (r < 0) ) ) {
            win++;
            if (r == 1)
                class_using++;
        }
        else
            cout << "predicted = " << c << " real = " << test_y[i] << endl;

    }

    cout << win << "/" << test_data_size << endl;
    cout << class_using << "/" << class_total;



    return 0;
}
