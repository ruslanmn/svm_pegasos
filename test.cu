#include <iostream>
#include <cstdlib>
#include <ctime>


using namespace std;

__global__ void parallelMultiply(float* a, float* b, int n, float* res) {
        int y = threadIdx.x;
		for(int column = 0; column < n; column++) {
			res[y*n+column] = 0;
			for(int i = 0; i < n; i++)
				res[y*n+column] += a[y*n + i] * b[i*n + column];
		}
}


void multiply(float* a, float* b, int n, float* res) {
	for(int y = 0; y < n; y++) {
		for(int column = 0; column < n; column++) {
			res[y*n+column] = 0;
			for(int i = 0; i < n; i++)
				res[y*n+column] += a[y*n + i] * b[i*n + column];
		}
	}
}

void generateMatrix(float* a, int n) {
	for(int i = 0; i < n*n; i++)
		a[i] = rand();
}

int main() {
	srand(time(0));
	int n = 250;
    size_t size = n*n*sizeof(float);
    clock_t begin_time;
    float* ha = (float*) malloc(sizeof(float) * n * n);
    generateMatrix(ha, n);
    float* hb = (float*) malloc(sizeof(float) * n * n);
    generateMatrix(hb, n);
    float* hres = (float*) malloc(sizeof(float) * n * n);

    begin_time = clock();
    multiply(ha, hb, n, hres);
    std::cout << float(clock () - begin_time)/CLOCKS_PER_SEC << endl;

    
    begin_time = clock();
    float* a;
    cudaMalloc(&a, size);
    cudaMemcpy(a, ha, size, cudaMemcpyHostToDevice);
    float* b;
    cudaMalloc(&b, size);
    cudaMemcpy(b, hb, size, cudaMemcpyHostToDevice);
    float* res;
    cudaMalloc(&res, size);
    cudaMemcpy(res, hres, size, cudaMemcpyHostToDevice);

    cudaFree(a);
    cudaFree(b);
    cudaFree(res);
    
    parallelMultiply<<<1, n>>>(a, b, n, res);
    
    std::cout << float(clock () - begin_time)/CLOCKS_PER_SEC << endl;

	return 0;
}

