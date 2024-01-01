CC=gcc
LDFLAGS= -llapacke -llapack -lcblas -lblas -lm -fopenmp
OFLAGS= -O2 -march=native -ftree-vectorize
DFLAGS= -Wall -Wextra -ggdb -g3
ASAN= -fsanitize=address -fno-omit-frame-pointer
INCLUDE= -I include -I deps/argtable

all :
	mkdir -p build2
	$(CC) $(DFLAGS) $(OFLAGS)  $(INCLUDE) src/main.c src/eig.c deps/argtable/argtable3.c -o build2/prr  $(LDFLAGS)

asan:
	mkdir -p build2
	$(CC) $(DFLAGS) $(OFLAGS)  $(ASAN) $(INCLUDE) src/main.c src/eig.c deps/argtable/argtable3.c -o build2/prr  $(LDFLAGS)
