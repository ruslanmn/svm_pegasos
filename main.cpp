#include <iostream>
#include "mnist_data_loader.h"
#include "svm.h"


using namespace std;




int main() {

    uint8_t **train_images, **test_images;
    uint8_t *train_labels, *test_labels;

    uint32_t* sizes = load_mnist_data("/home/svmfan/Desktop/images.data", "/home/svmfan/Desktop/labels.data",
                                      "/home/svmfan/Desktop/test-images.data", "/home/svmfan/Desktop/test-labels.data",
                    &train_images, &train_labels, &test_images, &test_labels);


    int win = 0;
    svm s(28*28);
    cout << "start fitting" << endl;
    s.fit(train_images, train_labels, sizes[0], 0.01, 60000);
    for(int i = 0; i < sizes[2]; i++) {
        double c = s.predict(test_images[i]);
        int r = -1;
        if (c >= 0)
            r = 1;

        if ( train_labels[i] == r )
            win++;
    }

    cout << win << "/" << sizes[2] << endl;
   /* int k;
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
    }*/

    return 0;
}
