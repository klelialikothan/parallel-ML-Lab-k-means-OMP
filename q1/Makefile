name = Parallel_ML_lab_2_1
src = $(wildcard *.cpp)
obj = $(src:/c=.o)

CC = g++
CFLAGS = -std=c++17 -O2 -pg -march=native -mtune=native
LIBFLAGS = -lm -fopt-info

compile: $(name)
$(name): $(obj)
	$(CC) $(CFLAGS) -o $@ $^ $(LIBFLAGS)

run:
	./$(name)
