CC=/usr/local/CUDA/bin/nvcc
TORCH=/usr/local
MATLAB_ROOT=/Applications/MATLAB_R2014a.app/
CFLAGS=-std=c++11 -I$(TORCH)/include -I$(TORCH)/include/TH -I./cunnproduction/

all: loader.o
	$(CC) -g -L$(TORCH)/lib $(CFLAGS)  -L. -lTH -lTHC -lcunnproduction test.cpp loader.o -o test
%.o: %.cpp
	$(CC) -c -Xcompiler -fPIC -o $@ $< $(CFLAGS)
clean:
	rm *.o test
