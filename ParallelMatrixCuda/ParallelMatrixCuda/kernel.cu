#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>
#include <cstdlib>

using namespace std;

const int matrixSize = 3300;

__global__ void matrixMult(const int* A, const int* B, int* C)
{
    int i0 = matrixSize * (blockDim.y * blockIdx.y + threadIdx.y);
    int j0 = blockDim.x * blockIdx.x + threadIdx.x;
    int sum = 0;

    for (int k = 0; k < matrixSize; k++)
        sum += A[i0 + k] * B[k * matrixSize + j0];

    int ind = matrixSize * (blockDim.y * blockIdx.y + threadIdx.y) + blockDim.x * blockIdx.x + threadIdx.x;
    C[ind] = sum;
}

int main(int argc, char** argv) {

    size_t size = matrixSize * matrixSize * sizeof(int);

    int* a = (int*)malloc(size);
    int* b = (int*)malloc(size);
    int* c = (int*)malloc(size);

    cout << "Filling matrixes" << endl;
    cout << endl;

    for (int i = 0; i < matrixSize; i++) {
        for (int j = 0; j < matrixSize; j++) {
            a[i * matrixSize + j] = rand() % 100;
            b[i * matrixSize + j] = rand() % 100;
        }
    }

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int* d_A = NULL;
    cudaMalloc((void**)&d_A, size);

    int* d_B = NULL;
    cudaMalloc((void**)&d_B, size);

    int* d_C = NULL;
    cudaMalloc((void**)&d_C, size);

    cudaMemcpy(d_A, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, b, size, cudaMemcpyHostToDevice);

    cout << "Starting calculations" << endl;
    cout << endl;

    int maxThreads = 4;

    for (int t = 1; t <= maxThreads; t++)
    {
        dim3 threadsPerBlock = dim3(t, t);
        dim3 blocksPerGrid = dim3(matrixSize / t, matrixSize / t);

        cudaEventRecord(start, 0);
        matrixMult<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C);

        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        float kernelTime;
        cudaEventElapsedTime(&kernelTime, start, stop);
        printf("Threads per block: %d; Blocks per grid: %d; KernelTime: %.2f seconds\n", t, matrixSize / t, kernelTime / 1000);

        cudaMemcpy(c, d_C, size, cudaMemcpyDeviceToHost);
    }

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(a);
    free(b);
    free(c);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
