#
# Works only for CCE
#
all: testpe dgetrftst_gpu

dgetrftst_gpu:   dgetrftst_gpu.o Wall_Time.o
	cc dgetrftst_gpu.o Wall_Time.o  -o dgetrftst_gpu

dgetrftst_gpu.o: dgetrftst_gpu.c
	cc  -fPIC -c  dgetrftst_gpu.c 

Wall_Time.o: Wall_Time.c
	cc  -fPIC -c Wall_Time.c

.PHONY: clean testpe
clean:
	rm -f *.o dgetrftst_gpu

testpe:
ifneq "$(PE_ENV)" "CRAY"
	@echo "Please swap to PrgEnv-cray and swap highest numbered version of cce module"
	@echo ""
endif
ifneq "$(CRAY_ACCEL_TARGET)" "nvidia20"
	@echo "Please run the following command: 'module load craype-accel-nvidia20'"
	@exit 1
endif
ifeq "$(CRAY_CUDATOOLKIT_VERSION)" ""
	@echo "Please run the following command: 'module load cudatoolkit'"
	@exit 1
endif
