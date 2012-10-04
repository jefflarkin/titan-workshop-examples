Titan Workshop Examples 
======================= 
Author: Jeff Larkin <larkin@cray.com> 
Some examples also from Keita Teranishi
------------------------------------- 

### Introduction 
These examples were put together for a Winter 2012 workshop at ORNL introducing the
Titan Cray XK6 system. They are designed as only simple examples of how to build
using various programming models and compilers available on that system using
the Cray build environment. Because the examples were devised during a time of
rapid development of the respective packages, these examples, expecially the
Makefiles, need to be made current from time to time. If you have issues, please
email me at the address above for support.

### Examples
* Example 1 demonstrates building a simple, single-node test case for CUDA for C
  and CUDA Fortran.
* Example 2 extends example 1 for building with MPI as well. Running the
  resulting executable with more than 1 process per node will give errors.
* Example 3 builds a simple OpenCL example. The Makefile must change certain
  flaps for support under different compilers. The following warning is expected
  when building using the PGI compiler
  > cc -ta=nvidia:4.1 -c -o openclExample.o openclExample.c PGC-W-0267-#warning
  > --   Need to implement some method to align data here
  > (/opt/nvidia/cudatoolkit/5.0.33.103/include/CL/cl_platform.h: 408)
  > PGC/x86-64 Linux 12.8-0: compilation completed with warnings
* Example 4 demonstrates calling libsci\_acc from a C codeand requires the Cray
  Compiler Environment (CCE)
* Example 5 demonstrates calling libsci\_acc from CUDA Fortran **(CURRENTLY BROKEN)**
* Example 6 demonstrates calling CUDA from OpenACC. It currently only supports
  the CCE.
* Example 7 demonstrates calling libsci_acc from OpenACC. **(WORKING BUT OUTDATED)**
* Example 7 demonstrates calling libsci_acc from PGIACC. **(CURRENTLY BROKEN)**
