CC=cc
OCLCC=cc
ifeq "$(PE_ENV)" "CRAY"
# CUDA works best with PrgEnv-cray with GNU interoperability enabled
CFLAGS += -hgnu 
OCLCC=nvcc
endif
ifeq "$(PE_ENV)" "PGI"
CFLAGS+=-ta=nvidia:4.1
OCLFLAGS+=-ta=nvidia:4.1
endif
opencl.exe: openclExample.o openclMain.o
	$(CC) $(OCLFLAGS) -o $@ $^ -lOpenCL

openclExample.o: openclExample.c
ifeq "$(CRAY_CUDATOOLKIT_VERSION)" ""
	@echo "Please run the following command: 'module load cudatoolkit'"
	@exit 1
endif
	$(OCLCC) $(OCLFLAGS) -c -o $*.o $<

.PHONY: clean
clean: 
	-rm -f *.o *.cub *.ptx *.lst *.mod opencl.exe

.SUFFIXES: .o .c 
.c.o:
	$(CC) -c $(CFLAGS) -o $*.o $<
