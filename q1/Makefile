name = Parallel_ML_lab_2_1
src = $(wildcard *.c)
obj = $(src:/c=.o)

CC = g++
CFLAGS = -std=c++17 -O2
LIBFLAGS = -lm

compile: $(name)
$(name): $(obj)
	$(CC) $(CFLAGS) -o $@ $^ $(LIBFLAGS)

run:
	./$(name)

clean:
	rm -f $(name)