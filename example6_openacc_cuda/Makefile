CC=cc
FC=ftn
CUDAC=nvcc
CUDACFLAGS=-arch=sm_20 --ptxas-options=-v

all: cudac.exe cudaf.exe

cudac.exe: scaleitC.o scaleitC_main.o
	${CC} -o $@ $^ -lcudart

cudaf.exe: scaleitC.o scaleitF_main.o
	$(FC) -o $@ $^ -lcudart

.PHONY: clean
clean: 
	-rm -f *.o *.cub *.ptx *.lst *.mod cudac.exe cudaf.exe cuda_profile_0.log

.SUFFIXES: .o .c .cu .F90
.c.o:
	$(CC) -c $(CFLAGS) -o $*.o $<
.cu.o:
	$(CUDAC) -c $(CUDACFLAGS) -o $*.o $<
.F90.o:
	$(FC) -c $(FFLAGS) -o $*.o $<
