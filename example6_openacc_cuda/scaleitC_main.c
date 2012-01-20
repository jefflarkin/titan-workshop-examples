#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

int scaleit_launcher_(double*, int*, int*);

int main(int argc, char **argv)
{
  int ierr, rank, n=16384, i;
  double *a;

  /* Allocate Array On Host */
  a = (double*)malloc(n*sizeof(double));

/* Allocate device array a. Copy data both to and from device. */
#pragma acc data copyout(a[0:n])
  {
#pragma acc parallel loop
    for(i=0; i<n; i++)
    {
      a[i] = i+1;
    }

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    /* Use device array when calling scaleit_launcher */
#pragma acc host_data use_device(a)
    {
      ierr = scaleit_launcher_(a, &n, &rank);
    }
  }

  for(i=0;i<n;i++)
  {
    if(a[i] != ((double)rank * (i+1)))
    {
      fprintf(stderr, "Error! %d: %lf\n",i,a[i]);
      return 1;
    }
  }
  fprintf(stdout, "Correct!\n");

  free(a);
  MPI_Finalize();
  return ierr;
}
