#include <CL/cl.h>
#include <stdlib.h>
#include <stdio.h>
/* 
** Based STRONGLY on oclLoadProgSource, which is provided by
** NVIDIA in the sdk.
*/
char* loadSource(const char *fname, size_t* sizeRead)
{
  FILE *f = NULL;
  size_t len;
  char *src;

  f = fopen(fname, "r");
  if(f == NULL )
    return NULL;
  /* get source length */
  fseek(f, 0, SEEK_END);
  len = ftell(f);
  fseek(f, 0, SEEK_SET);

  src = calloc(len+1, sizeof(char));

  if(fread(src, len, 1, f) != 1)
  {
    fclose(f);
    free(src);
    return 0;
  }

  fclose(f);
  *sizeRead = len + 1;

  return src;
}

// OpenCL Resources
cl_context context;        // OpenCL context
cl_command_queue queue;    // OpenCL command que
cl_platform_id cpPlatform; // OpenCL platform
cl_device_id cdDevice;     // OpenCL device
cl_program cpProgram;      // OpenCL program
cl_kernel ckKernel;        // OpenCL kernel

/*
** OpenCL compiles at runtime, which is very expensive.  This function
** does to compilation once and saves is globally for reuse.
*/
void setupocl_()
{
  char* cSourceCL = NULL;         // Buffer to hold source for compilation
  size_t szKernelLength;          // Byte size of kernel code
  int err;
  char errstr[1024000];

  err = clGetPlatformIDs(1, &cpPlatform, NULL);
  if (err != CL_SUCCESS) 
  {
    fprintf(stderr, "Failed to get platforms. %d\n", err);
    return;
  } 

  err = clGetDeviceIDs(cpPlatform, CL_DEVICE_TYPE_GPU, 1, &cdDevice, NULL);
  if (err != CL_SUCCESS) 
  {
    fprintf(stderr, "Failed to get devices. %d\n", err);
    return;
  } 

  /*context = clCreateContextFromType(NULL, CL_DEVICE_TYPE_GPU, NULL, NULL,
  ** &err);*/
  context = clCreateContext(0, 1, &cdDevice, NULL, NULL, &err);
  if (err != CL_SUCCESS) 
  {
    fprintf(stderr, "Failed to create context. %d\n", err);
    return;
  } 
  /*queue =  clCreateCommandQueue(context, NULL, 0, &err);*/
  queue = clCreateCommandQueue(context, cdDevice, 0, &err);
  if (err != CL_SUCCESS) 
  {
    fprintf(stderr, "Failed to create queue. %d\n", err);
    return;
  }

  cSourceCL = loadSource("scaleit.ocl", &szKernelLength);
  cpProgram = clCreateProgramWithSource(context, 1, (const char **)&cSourceCL, &szKernelLength, &err);
  if (err != CL_SUCCESS) 
  {
    fprintf(stderr, "Failed to create program. %d\n", err);
    return;
  } 

  err = clBuildProgram(cpProgram, 0, NULL, NULL, NULL, NULL);
  if (err != CL_SUCCESS) 
  {
   fprintf(stderr, "Failed to build program.\n");
   err = clGetProgramBuildInfo(cpProgram, cdDevice,
                               CL_PROGRAM_BUILD_LOG,
                               1024000, errstr, &szKernelLength);
   fprintf(stderr,"Err: %s\n", errstr);
   return;
  } 

  ckKernel = clCreateKernel(cpProgram, "scaleIt", &err);
  if (err != CL_SUCCESS) 
  {
    fprintf(stderr, "Failed to create kernel. %d\n", err);
    return;
  } 

}
/*
** Release the resources form setupocl
*/
void releaseocl_()
{
  clReleaseKernel(ckKernel);
  clReleaseCommandQueue(queue);
  clReleaseProgram(cpProgram);
  clReleaseContext(context);
}

void oclScaleIt(double *a,
                int n,
                int scaleBy)
{
  size_t szGlobalWorkSize[3];  // var for Total # of work items
  size_t szLocalWorkSize[3];   // var for # of work items in the work group	
  int err;

  cl_mem a_d;

  setupocl_();

  a_d = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_double) *n, NULL, &err);
  if (err != CL_SUCCESS) 
  {
    fprintf(stderr, "Failed to create buffer. %d line: %d\n", err, __LINE__);
    return;
  } 

  err  = clEnqueueWriteBuffer(queue, a_d, CL_FALSE, 0, sizeof(cl_double) * n, a, 0, NULL, NULL);
  if (err != CL_SUCCESS) 
  {
    fprintf(stderr, "Failed to copy buffer. %d\n", err);
    return;
  } 

  err  = clSetKernelArg(ckKernel, 0, sizeof(cl_double), (void*)&a_d);
  err |= clSetKernelArg(ckKernel, 1, sizeof(cl_int), (void*)&scaleBy);
  if (err != CL_SUCCESS) 
  {
    fprintf(stderr, "Failed to set argument. %d\n", err);
    return;
  } 
  szGlobalWorkSize[0] = n;
  szGlobalWorkSize[1] = 1;
  szGlobalWorkSize[2] = 1;
  szLocalWorkSize[0] = 1024;
  szLocalWorkSize[1] = 1;
  szLocalWorkSize[2] = 1;

  err = clEnqueueNDRangeKernel(queue, ckKernel, 3, NULL, szGlobalWorkSize, szLocalWorkSize, 0, NULL, NULL);
  if (err != CL_SUCCESS) 
  {
    fprintf(stderr, "Failed to enqueue kernel %d.\n", err);
    return;
  }

  err = clEnqueueReadBuffer(queue, a_d, CL_TRUE, 0, sizeof(cl_double) * n, a, 0, NULL, NULL);
  if (err != CL_SUCCESS) 
  {
    fprintf(stderr, "Failed to copy buffer back. %d\n", err);
    return;
  } 
  clFinish(queue); // Should be superfluous b/c of blocking copy

  // Release the resources
  clReleaseMemObject(a_d);

  releaseocl_();
}
