#include <mpi.h>
#include <iostream>

int main() {
    MPI_Init(NULL, NULL);
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    std::cout << "Hello from rank " << rank << std::endl;
    MPI_Finalize();
    return 0;
}
