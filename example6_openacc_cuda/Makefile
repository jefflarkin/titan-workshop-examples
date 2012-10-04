CC=cc
FC=ftn
CUDAC=nvcc
CUDACFLAGS=-arch=sm_20 --ptxas-options=-v
ifeq "$(PE_ENV)" "PGI"
#PGI OpenACC support does not yet support this demo
CFLAGS+=-ta=nvidia:4.1 -acc
FFLAGS+=-ta=nvidia:4.1 -acc
LDFLAGS+=-ta=nvidia:4.1 -acc
endif

all: testpe cudac.exe cudaf.exe

cudac.exe: scaleitC.o scaleitC_main.o
	$(CC) -o $@ $^ -lcudart $(LDFLAGS)

cudaf.exe: scaleitC.o scaleitF_main.o
	$(FC) -o $@ $^ -lcudart $(LDFLAGS)


.PHONY: clean testpe
clean: 
	-rm -f *.o *.cub *.ptx *.lst *.mod cudac.exe cudaf.exe cuda_profile_0.log
testpe:
ifneq "$(PE_ENV)" "CRAY"
	@echo "Please swap to PrgEnv-cray"
	@exit 1
endif

.SUFFIXES: .o .c .cu .F90
.c.o:
	$(CC) -c $(CFLAGS) -o $*.o $<
.cu.o:
	$(CUDAC) -c $(CUDACFLAGS) -o $*.o $<
.F90.o:
	$(FC) -c $(FFLAGS) -o $*.o $<
