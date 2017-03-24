#include <iostream>
#include "mnist_data_loader.h"

using namespace std;

int main() {

    uint8_t **train_images, **test_images;
    uint8_t *train_labels, *test_labels;

    uint32_t* sizes = load_mnist_data("/home/svmfan/Desktop/images.data", "/home/svmfan/Desktop/labels.data",
                                      "/home/svmfan/Desktop/test-images.data", "/home/svmfan/Desktop/test-labels.data",
                    &train_images, &train_labels, &test_images, &test_labels);

    int k;
    for(int i = 0; i < sizes[0]; i++) {
        for(int y = 0; y < 28; y++ ) {
            for (int x = 0; x < 28; x++) {
                uint8_t p = train_images[i][y * 28 + x];
                if( p == 0 )
                    cout << " ";
                else
                    cout << "x";
            }
            cout << endl;
            cout << train_labels[i];
            cout << endl;
        }
        cin >> k;
    }

    return 0;
}
