.PHONY: clean all 

DIR := ${CURDIR}

SDIR = $(DIR)/src
ODIR = $(DIR)/obj
LDIR = $(DIR)/lib
IDIR = $(DIR)/include
CDIR = $(DIR)/cython
PDIR = $(DIR)/optweight

#NEWDIRS = $(LDIR)
NEWDIRS = $(LDIR) ${ODIR}
$(info $(shell mkdir -p -v $(NEWDIRS)))

CFLAGS = -g -Wall -std=c99 -pedantic
OMPFLAG = -fopenmp
OPTFLAG = -O3 -ffast-math

# ADDDED
#MKLROOT := /opt/intel/compilers_and_libraries_2020.1.217/linux/mkl/
LINK_MKL = -L$(MKLROOT)/lib/intel64 -Wl,--no-as-needed -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lpthread
LINK_COMMON = -lm


all: $(LDIR)/liboptweight.so 

#$(LDIR)/liboptweight.so:
#	$(CC) $(CFLAGS) $(OMPFLAG) $(OPTFLAG) -shared -o $(LDIR)/liboptweight_c_utils.so -fPIC ${SDIR}/optweight_alm_c_utils.c -I${IDIR}
$(LDIR)/liboptweight.so: $(ODIR)/optweight_alm_c_utils.o $(ODIR)/optweight_mat_c_utils.o
	$(CC) -shared -o $(LDIR)/liboptweight_c_utils.so -fPIC ${ODIR}/optweight_alm_c_utils.o ${ODIR}/optweight_mat_c_utils.o -I${IDIR} $(LINK_COMMON) $(LINK_MKL) -lgomp

$(ODIR)/optweight_alm_c_utils.o:
	$(CC) $(CFLAGS) $(OMPFLAG) $(OPTFLAG) -c -o $@ $< ${SDIR}/optweight_alm_c_utils.c -I${IDIR} -fPIC

$(ODIR)/optweight_mat_c_utils.o:
	$(CC) $(CFLAGS) $(OMPFLAG) $(OPTFLAG) -c -o $@ $< ${SDIR}/optweight_mat_c_utils.c $(LINK_COMMON) $(LINK_MKL) -lgomp -fPIC

clean:
	rm -rf $(LDIR)
	rm -rf $(ODIR)
	rm -f $(CDIR)/*.c
	rm -f $(PDIR)/*.so
