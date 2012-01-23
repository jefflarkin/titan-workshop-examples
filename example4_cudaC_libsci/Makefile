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
	@echo "Please run the following command: 'module load xtpe-accel-nvidia20'"
	@exit 1
endif
ifeq "$(CRAY_CUDA_VERSION)" ""
	@echo "Please run the following command: 'module load cuda'"
	@exit 1
endif
ifneq "$(ASYNCPE_VERSION)" "5.05"
	@echo "Please run the following command: 'module swap xt-asyncpe xt-asyncpe/5.05'"
	@exit 1
endif
