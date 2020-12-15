name = Parallel_ML_lab_2_3
src = $(wildcard *.cpp)
obj = $(src:/c=.o)

CC = g++
CFLAGS = -std=c++17 -O2 -march=native -mtune=native
LIBFLAGS = -lm -fopenmp -fopt-info

compile: $(name)
$(name): $(obj)
	$(CC) $(CFLAGS) -o $@ $^ $(LIBFLAGS)

run:
	./$(name)

clean:
	rm -f $(name)
