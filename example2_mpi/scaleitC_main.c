#include <mpi.h>
#include <stdio.h>

/* 
 * Cray's cc wrapper doesn't understand CUDA, nvcc doesn't
 * understand Cray's MPI environment, so all CUDA-related
 * items are isolated into the launcher and all MPI are 
 * isolated here.
 */
int scaleit_launcher(int);

int main(int argc, char **argv)
{
  int ierr, rank;

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  ierr = scaleit_launcher(rank);

  MPI_Finalize();
  return ierr;
}
