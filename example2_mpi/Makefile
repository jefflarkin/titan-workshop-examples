CC=cc
FC=ftn
CUDAC=nvcc
CUDACFLAGS=-arch=sm_20 --ptxas-options=-v
CUDAF=${FC}
CUDAFFLAGS=-Mcuda=cc20,ptxinfo,cuda4.1 -ta=nvidia:cc20,4.1,time -Minfo=all
ifeq "$(PE_ENV)" "PGI"
LDFLAGS+=-ta=nvidia:4.1
CFLAGS+=-ta=nvidia:4.1
endif

all: cudaf.exe cudac.exe

cudaf.exe: scaleitF.o
ifeq "$(CRAY_CUDATOOLKIT_VERSION)" ""
	@echo "Please run the following command: 'module load cudatoolkit'"
	@exit 1
endif
ifneq "$(PE_ENV)" "PGI"
	@echo "Please swap to PrgEnv-pgi"
	@exit 1
endif
	${CUDAF} -o $@ ${CUDAFFLAGS} $<

cudac.exe: scaleitC.o scaleitC_main.o
ifeq "$(CRAY_CUDATOOLKIT_VERSION)" ""
	@echo "Please run the following command: 'module load cudatoolkit'"
	@exit 1
endif
	${CC} -o $@ $^ $(LDFLAGS)

.PHONY: clean
clean: 
	-rm -f *.o *.cub *.ptx *.lst *.mod cudaf.exe cudac.exe

.SUFFIXES: .o .F90 .c .cu
.F90.o:
	$(FC) -c $(CUDAFFLAGS) $(FFLAGS) -o $*.o $<
.c.o:
	$(CC) -c $(CFLAGS) -o $*.o $<
.cu.o:
	$(CUDAC) -c $(CUDACFLAGS) -o $*.o $<
