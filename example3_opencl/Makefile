CC=cc
opencl.exe: openclExample.o openclMain.o
	${CC} -o $@ $^ -lOpenCL

.PHONY: clean
clean: 
	-rm -f *.o *.cub *.ptx *.lst *.mod opencl.exe

.SUFFIXES: .o .c 
.c.o:
	$(CC) -c $(CFLAGS) -o $*.o $<
