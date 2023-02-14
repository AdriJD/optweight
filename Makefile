.PHONY: clean all 

DIR := ${CURDIR}

SDIR = $(DIR)/src
ODIR = $(DIR)/obj
LDIR = $(DIR)/lib
IDIR = $(DIR)/include
CDIR = $(DIR)/cython
PDIR = $(DIR)/optweight

NEWDIRS = $(LDIR) ${ODIR}
$(info $(shell mkdir -p -v $(NEWDIRS)))

CFLAGS = -g -Wall -Wextra -std=c99 -pedantic
OMPFLAG = -fopenmp
OPTFLAG = -O3 -ffast-math -march=native

# We explicitely link to the sequential version of MKL, because we want to avoid nested parallelization.
# We will do the openMP threading ourselves in the outer loops of the C scripts.
LINK_MKL = -L${MKLROOT}/lib/intel64 -lmkl_rt -Wl,--no-as-needed -lpthread -lm -ldl
CFLAGS_MKL = -m64  -I"${MKLROOT}/include"

all: $(LDIR)/liboptweight.so 

$(LDIR)/liboptweight.so: $(ODIR)/optweight_alm_c_utils.o $(ODIR)/optweight_mat_c_utils.o
	$(CC) ${OMPFLAG} -shared -o $(LDIR)/liboptweight_c_utils.so -fPIC ${ODIR}/optweight_alm_c_utils.o ${ODIR}/optweight_mat_c_utils.o -I${IDIR} $(LINK_MKL) -lgomp

$(ODIR)/optweight_alm_c_utils.o:
	$(CC) $(CFLAGS) $(OMPFLAG) $(OPTFLAG) -c -o $@ $< ${SDIR}/optweight_alm_c_utils.c -I${IDIR} -fPIC

$(ODIR)/optweight_mat_c_utils.o:
	$(CC) $(CFLAGS) $(CFLAGS_MKL) $(OMPFLAG) $(OPTFLAG) -c -o $@ $< ${SDIR}/optweight_mat_c_utils.c $(LINK_MKL) -lgomp -I${IDIR} -fPIC

clean:
	rm -rf $(LDIR)
	rm -rf $(ODIR)
	rm -f $(CDIR)/*.c
	rm -f $(PDIR)/*.so
