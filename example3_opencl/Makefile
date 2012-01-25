CC=cc
OCLCC=cc
ifeq "$(PE_ENV)" "CRAY"
# CUDA works best with PrgEnv-cray with GNU interoperability enabled
CFLAGS += -hgnu 
OCLCC=nvcc
endif
opencl.exe: openclExample.o openclMain.o
	${CC} -o $@ $^ -lOpenCL

openclExample.o: openclExample.c
ifeq "$(CRAY_CUDA_VERSION)" ""
	@echo "Please run the following command: 'module load cuda'"
	@exit 1
endif
	$(OCLCC) -c -o $*.o $<

.PHONY: clean
clean: 
	-rm -f *.o *.cub *.ptx *.lst *.mod opencl.exe

.SUFFIXES: .o .c 
.c.o:
	$(CC) -c $(CFLAGS) -o $*.o $<
