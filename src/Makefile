CC=g++

PATH_TO_EIGEN=path/to/eigen/parent/directory/eigen
PATH_TO_LBFGS=path/to/LBFGSpp/parent/directory/LBFGSpp/include

CFLAGS=-I. -I$(PATH_TO_EIGEN) -I$(PATH_TO_LBFGS) -fopenmp -O3 -march=native
LFLAGS=


./PHONY: evaluate
evaluate: evaluate.o matrix_builder.o
	$(CC) $(CFLAGS) -o $@ $^ $(LFLAGS)

evaluate.o: evaluate.cpp matrix_builder.h types.h
	$(CC) $(CFLAGS) -c $^

./PHONY: optimize
optimize: optimize.o matrix_builder.o
	$(CC) $(CFLAGS) -o $@ $^ $(LFLAGS)

optimize.o: optimize.cpp matrix_builder.h types.h
	$(CC) $(CFLAGS) -c $^

matrix_builder.o: matrix_builder.cpp matrix_builder.h types.h
	$(CC) $(CFLAGS) -c $^


./PHONY: clean
clean:
	rm -f evaluate optimize *.o *.gch
