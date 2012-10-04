all: testpe fortran_example c_example

fortran_example: fortran_example.o Wall_Time.o
	ftn -o fortran_example fortran_example.o Wall_Time.o

c_example: c_example.o Wall_Time.o
	cc -o c_example c_example.o Wall_Time.o

Wall_Time.o : Wall_Time.c
	cc -c Wall_Time.c

fortran_example.o: fortran_example.f90 
	ftn -c fortran_example.f90

c_example.o: c_example.c
	cc -c c_example.c

.PHONY: clean testpe
clean:
	rm -f *.o  fortran_example c_example

testpe:
ifneq "$(PE_ENV)" "CRAY"
	@echo "Please swap to PrgEnv-cray"
	@exit 1
endif
ifeq "$(CRAY_ACCEL_TARGET)" ""
	@echo "Please run the following command (Kepler): 'module load craype-accel-nvidia35'"
	@echo "Please run the following command (Fermi): 'module load craype-accel-nvidia20'"
	@exit 1
endif
