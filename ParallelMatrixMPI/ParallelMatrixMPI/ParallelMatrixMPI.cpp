#include <iostream>
#include <cstdlib>
#include "mpi.h"

using namespace std;

int main(int argc, char** argv) {
    srand(time(NULL));

    int i, j, k, l;
    int* a, * b, * c, * buffer, * ans;
    int matrixSize = 3300;
    int rank, numberOfProcesses, line;

    MPI_Init(NULL, NULL);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numberOfProcesses);

    line = matrixSize / numberOfProcesses;
    a = (int*)malloc(sizeof(int) * matrixSize * matrixSize);
    b = (int*)malloc(sizeof(int) * matrixSize * matrixSize);
    c = (int*)malloc(sizeof(int) * matrixSize * matrixSize);
    buffer = (int*)malloc(sizeof(int) * matrixSize * line);
    ans = (int*)malloc(sizeof(int) * matrixSize * line);

    if (rank == 0) {
        double start, stop;

        cout << "Filling matrixes" << endl;
        cout << endl;
        for (i = 0; i < matrixSize; i++) {
            for (j = 0; j < matrixSize; j++) {
                a[i * matrixSize + j] = rand() % 100;
                b[i * matrixSize + j] = rand() % 100;
            }
        }
        cout << "Starting calculations" << endl;
        cout << endl;

        start = MPI_Wtime();
        
        for (i = 1; i < numberOfProcesses; i++) {
            MPI_Send(b, matrixSize * matrixSize, MPI_INT, i, 0, MPI_COMM_WORLD);
        }
        
        for (l = 1; l < numberOfProcesses; l++) {
            MPI_Send(a + (l - 1) * line * matrixSize, matrixSize * line, MPI_INT, l, 1, MPI_COMM_WORLD);
        }
        
        for (k = 1; k < numberOfProcesses; k++) {
            MPI_Recv(ans, line * matrixSize, MPI_INT, k, 3, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            
            for (i = 0; i < line; i++) {
                for (j = 0; j < matrixSize; j++) {
                    c[((k - 1) * line + i) * matrixSize + j] = ans[i * matrixSize + j];
                }
            }
        }
        
        for (i = (numberOfProcesses - 1) * line; i < matrixSize; i++) {
            for (j = 0; j < matrixSize; j++) {
                int temp = 0;
                for (k = 0; k < matrixSize; k++)
                    temp += a[i * matrixSize + k] * b[k * matrixSize + j];
                c[i * matrixSize + j] = temp;
            }
        }
        
        stop = MPI_Wtime();

        printf("Processes: %d; Time: %d seconds\n", numberOfProcesses, (int)(stop - start));

        free(a);
        free(b);
        free(c);
        free(buffer);
        free(ans);
    }

    else {
        MPI_Recv(b, matrixSize * matrixSize, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        MPI_Recv(buffer, matrixSize * line, MPI_INT, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
       
        for (i = 0; i < line; i++) {
            for (j = 0; j < matrixSize; j++) {
                int temp = 0;
                for (k = 0; k < matrixSize; k++)
                    temp += buffer[i * matrixSize + k] * b[k * matrixSize + j];
                ans[i * matrixSize + j] = temp;
            }
        }
        
        MPI_Send(ans, line * matrixSize, MPI_INT, 0, 3, MPI_COMM_WORLD);
    }

    MPI_Finalize();

    return 0;
}
