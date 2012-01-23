CC=cc
FC=ftn
CUDAC=nvcc
CUDACFLAGS=-arch=sm_20 --ptxas-options=-v
CUDAF=${FC}
CUDAFFLAGS=-Mcuda=cc20,ptxinfo,cuda3.2 -ta=nvidia:cc20,time -Minfo=all
ifeq "$(PE_ENV)" "PGI"
LDFLAGS+=-lcudart
endif

all: cudaf.exe cudac.exe

cudaf.exe: scaleitF.o
ifeq "$(CRAY_CUDA_VERSION)" ""
	@echo "Please run the following command: 'module load cuda'"
	@exit 1
endif
ifneq "$(PE_ENV)" "PGI"
	@echo "Please swap to PrgEnv-pgi"
	@exit 1
endif
	${CUDAF} -o $@ ${CUDAFFLAGS} $<

cudac.exe: scaleitC.o scaleitC_main.o
ifeq "$(CRAY_CUDA_VERSION)" ""
	@echo "Please run the following command: 'module load cuda'"
	@exit 1
endif
ifeq "$(PE_ENV)" "CRAY"
	@echo "If you receive a link error for -lsci_acc, please ensure xtpe-accel-nvidia20 is loaded and swap to xt-asyncpe/5.05 (even if it is already loaded)."
	@echo ""
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
