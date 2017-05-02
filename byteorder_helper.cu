#include <stdint.h>
#include <cstdlib>
#include <cstdio>
//
// Created by svmfan on 3/25/17.
//

void flip(uint8_t* data, unsigned int size) {
    for(unsigned int off = 0; off < size / 2; off++) {
        unsigned int twin = size - off - 1;
        data[off] ^= data[twin];
        data[twin] ^= data[off];
        data[off] ^= data[twin];
    }
}

void fread_uint32_with_flip(uint32_t* i, FILE* f) {
    fread(i, 4, 1, f);
    flip((uint8_t*)i, 4);
}