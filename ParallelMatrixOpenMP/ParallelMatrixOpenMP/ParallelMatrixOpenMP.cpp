#include <iostream>
#include <cstdlib>
#include <omp.h>

using namespace std;

int main(int argc, char** argv) {
    
    int matrixSize = 3300;
    int* a, * b, * c;

    a = (int*)malloc(sizeof(int) * matrixSize * matrixSize);
    b = (int*)malloc(sizeof(int) * matrixSize * matrixSize);
    c = (int*)malloc(sizeof(int) * matrixSize * matrixSize);
    
    cout << "Filling matrixes" << endl;
    cout << endl;

    for (int i = 0; i < matrixSize; i++) {
        for (int j = 0; j < matrixSize; j++) {
            a[i * matrixSize + j] = rand() % 100;
            b[i * matrixSize + j] = rand() % 100;
        }
    }

    srand(time(NULL));

    cout << "Starting calculations" << endl;
    cout << endl;

    int maxThreads = 4;

    for (int t = 1; t <= maxThreads; t++)
    {
        omp_set_num_threads(t);
        int i, j, k;
        unsigned int start = clock();
#pragma omp parallel for private(i, j, k)
        for (i = 0; i < matrixSize; i++) {
            for (j = 0; j < matrixSize; j++) {
                for (k = 0; k < matrixSize; k++) {
                    c[i * matrixSize + j] += (a[i * matrixSize + k] * b[k * matrixSize + j]);
                }
            }
        }
        printf("Threads: %d; Time: %d seconds\n", t, (clock() - start) / 1000);
    }

    free(a);
    free(b);
    free(c);

    return 0;
}
