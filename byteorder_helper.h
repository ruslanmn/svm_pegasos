//
// Created by svmfan on 3/25/17.
//

#ifndef SVM_BYTE_HELPER_H
#define SVM_BYTE_HELPER_H

#include <cstdint>
#include <cstdlib>
#include <cstdio>


void flip(uint8_t* data, size_t size);
void fread_uint32_with_flip(uint32_t* i, FILE* f);

#endif //SVM_BYTE_HELPER_H
