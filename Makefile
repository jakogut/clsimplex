CC = gcc
LD = $(CC)

CFLAGS = -std=gnu11 -Wall -Wextra -Werror -pedantic -pipe -march=native -g -fopenmp -static
OFLAGS =
LFLAGS = -lm -lc -lOpenCL -lpthread -fopenmp

OPTIMIZATION = -Ofast

CFLAGS += $(OPTIMIZATION)

OBJECTS = cl_common.o

all: clsimplex

clsimplex: clsimplex.o $(OBJECTS)
	$(LD) $< $(OBJECTS) $(LFLAGS) -o clsimplex

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -rf clsimplex gmon.out *.save *.o core* vgcore*

rebuild: clean all

.PHONY : clean
.SILENT : clean
.NOTPARALLEL : clean
