#include <cuda.h>
#include <stdio.h>

__global__
void scaleit_kernel(double *a,int n, int scaleBy)
{
  /* Determine my index */
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < n) 
  {
    a[i] = a[i] * (double)scaleBy;
  }
}

extern "C" {
int scaleit_launcher(int scaleBy)
{
  double *h_a, *d_a;
  int i,n=16384;
  dim3 block, grid;

  /* Allocate Host Pointer */
  h_a = (double*)malloc(n*sizeof(double));
  for(i=0; i<n; i++)
  {
    h_a[i] = i+1;
  }

  /* Allocate Device Pointer */
  cudaMalloc((void**)&d_a, n*sizeof(double));
  if ( d_a == NULL )
  {
    fprintf(stderr,"Failed to malloc!\n");
    exit(1);
  }

  /* Decompose Problem */
  block = dim3(1024, 1, 1);
  grid = dim3(n/block.x, 1, 1);

  /* Copy from Host to Device */
  cudaMemcpy(d_a, h_a, n*sizeof(double),cudaMemcpyHostToDevice);

  /* Launch Compute Kernel */
  scaleit_kernel<<<grid,block>>>(d_a,n,scaleBy);

  /* Copy from Device to Host */
  cudaMemcpy(h_a, d_a, n*sizeof(double),cudaMemcpyDeviceToHost);

  for(i=0;i<n;i++)
  {
    if(h_a[i] != ((double)scaleBy * (i+1)))
    {
      fprintf(stderr, "Error! %d: %lf\n",i,h_a[i]);
      return 1;
    }
  }
  fprintf(stdout, "Correct!\n");

  /* Free Device Pointer */
  cudaFree(d_a);

  /* Free Host Pointer */
  free(h_a);

  return 0;
}
}
