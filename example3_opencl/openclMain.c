#include <stdlib.h>
#include <stdio.h>
#include <mpi.h>
void oclScaleIt(double *a,
                int n,
                int scaleBy);

int main(int argc, char **argv)
{
  double *a;
  int i,rank, n=16384;
  MPI_Init(&argc,&argv);
  MPI_Comm_rank(MPI_COMM_WORLD,&rank);

  a = (double*)malloc(n*sizeof(double));

  for(i=0;i<n;i++)
  {
    a[i] = i+1;
  }

  oclScaleIt(a,n,rank);

  for(i=0;i<n;i++)
  {
    if(a[i] != ((double)rank * (i+1))) 
    {
      fprintf(stderr, "[%d] Error! %d: %lf\n",rank,i,a[i]);
      MPI_Abort(MPI_COMM_WORLD,1);
    }
  }
  printf("Correct!\n");

  free(a);
  MPI_Finalize();
  return 0;
}
