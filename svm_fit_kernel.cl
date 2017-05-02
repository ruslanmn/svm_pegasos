float kernel_func(__global float* x, __global float* b, size_t size) {
    float s = 0;
    while( size > 0 ) {
        size--;
        s += x[size] * b[size];
    }

    return s;
}


__kernel void fit(__global uint* indices, uint T, __global uint* a, __global float* x, uint weight_size, __global float* y, uint data_size, float h) {
    printf("enter");
    uint id = get_global_id(0);
    if (id >= T)
        return;

    uint t =  id + 1;
    uint i = indices[id];

    float q = 1/(h*t);
    float s = 0;

    for(int j = 0; j < data_size; j++)
     if( j != i ) {
         s += a[j] * y[j] * kernel_func(&x[i * weight_size], &x[j * weight_size], weight_size);
     }

    if (y[i] * q * s < 1) {
     atomic_add(&a[i], 1);
    }
}