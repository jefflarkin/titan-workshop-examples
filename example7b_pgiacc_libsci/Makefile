all: fortran_example c_example

fortran_example: fortran_example.o Wall_Time.o
	ftn -ta=nvidia -o fortran_example fortran_example.o Wall_Time.o -Mcuda

c_example: c_example.o Wall_Time.o
	cc -ta=nvidia -o c_example c_example.o Wall_Time.o

Wall_Time.o : Wall_Time.c
	cc -c Wall_Time.c

fortran_example.o: fortran_example.f90 
	ftn -ta=nvidia -c fortran_example.f90 -Mcuda

c_example.o: c_example.c
	cc -ta=nvidia -c c_example.c

.PHONY: clean 
clean:
	rm -f *.o  fortran_example c_example

