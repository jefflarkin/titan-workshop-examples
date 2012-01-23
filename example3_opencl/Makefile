CC=cc
ifeq "$(PE_ENV)" "CRAY"
# CUDA works best with PrgEnv-cray with GNU interoperability enabled
CFLAGS += -hgnu 
endif
opencl.exe: openclExample.o openclMain.o
ifeq "$(CRAY_CUDA_VERSION)" ""
	@echo "Please run the following command: 'module load cuda'"
	@exit 1
endif
ifneq "$(PE_ENV)" "GNU"
	@echo "Please swap to PrgEnv-gnu or PrgEnv-cray"
	@exit 1
endif
	${CC} -o $@ $^ -lOpenCL

.PHONY: clean
clean: 
	-rm -f *.o *.cub *.ptx *.lst *.mod opencl.exe

.SUFFIXES: .o .c 
.c.o:
	$(CC) -c $(CFLAGS) -o $*.o $<
