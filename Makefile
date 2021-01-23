.PHONY: clean all 

DIR := ${CURDIR}

SDIR = $(DIR)/src
LDIR = $(DIR)/lib
IDIR = $(DIR)/include
CDIR = $(DIR)/cython
PDIR = $(DIR)/optweight

HS_SDIR = $(DIR)/hyperspherical/src
HS_IDIR = $(DIR)/hyperspherical/include

NEWDIRS = $(LDIR)
$(info $(shell mkdir -p -v $(NEWDIRS)))

CFLAGS = -g -Wall -std=c99 -pedantic
OMPFLAG = -fopenmp
OPTFLAG = -O3 -ffast-math

all: $(LDIR)/liboptweight.so 

$(LDIR)/liboptweight.so:
	$(CC) $(CFLAGS) $(OMPFLAG) $(OPTFLAG) -shared -o $(LDIR)/liboptweight_c_utils.so -fPIC ${SDIR}/optweight_alm_c_utils.c -I${IDIR}

clean:
	rm -rf $(LDIR)
	rm -f $(CDIR)/*.c
	rm -f $(PDIR)/*.so
