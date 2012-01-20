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

/* nvcc uses C++ name mangling by default */
extern "C"
{
  int scaleit_launcher_(double *d_a, int *n, int *scaleBy)
  {
    dim3 block, grid;
  
    /* Decompose Problem */
    block = dim3(1024, 1, 1);
    grid = dim3(*n/block.x, 1, 1);
  
    /* Launch Compute Kernel */
    scaleit_kernel<<<grid,block>>>(d_a,*n,*scaleBy);
  
    return 0;
  }
}
