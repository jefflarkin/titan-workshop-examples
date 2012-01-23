matmul_1: matmul_1.CUF
ifneq "$(PE_ENV)" "PGI"
	@echo "Please swap to PrgEnv-pgi"
	@echo "After swapping, you may need to run : 'module load libsci_acc' and"
	@echo "'module swap xt-asyncpe xt-asyncpe/5.05' if you get linker errors."
	@exit 1
endif
	ftn -o matmul_1 matmul_1.CUF
.PHONY:
clean:
	rm -f *.o matmul_1 *.mod
