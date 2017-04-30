double kernel_func(__global double* x, __global double* b, size_t size) {
    double s = 0;
    while( size > 0 ) {
        size--;
        s += x[size] * b[size];
    }

    return s;
}


__kernel void fit(__global uint* indices, uint T, __global uint* a, __global double* x, uint weight_size, __global double* y, uint data_size, double h) {
    uint id = get_global_id(0);

    if (id >= T)
        return;

    uint t =  id + 1;
    uint i = indices[id];

    double q = 1/(h*t);
    double s = 0;

    for(int j = 0; j < data_size; j++)
     if( j != i ) {
         s += a[j] * y[j] * kernel_func(&x[i * weight_size], &x[j * weight_size], weight_size);
     }

    if (y[i] * q * s < 1) {
     atomic_add(&a[i], 1);
    }
}
