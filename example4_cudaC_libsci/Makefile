#
# Works only for CCE
#
all: dgetrftst_gpu

dgetrftst_gpu:   dgetrftst_gpu.o Wall_Time.o
	cc dgetrftst_gpu.o Wall_Time.o  -o dgetrftst_gpu

dgetrftst_gpu.o: dgetrftst_gpu.c
	cc  -fPIC -c  dgetrftst_gpu.c 

Wall_Time.o: Wall_Time.c
	cc  -fPIC -c Wall_Time.c


clean:
	rm -f *.o dgetrftst_gpu

