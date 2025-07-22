#CC := gcc

#CFLAGS := -mavx -mfma -fopenmp

#COMPILE_COMMAND := $(CC) $(CFLAGS)

SRC_DIR := ./src
LIB_DIR := ./lib
GEN_DIR := ./gencode

VERSIONS := sgi sgi_mpi sgi_omp sgi_linearc sgi_linear

.PHONY: all
all:
	for ver in $(VERSIONS); do \
		$(MAKE) $$ver; \
	done

.PHONY: $(VERSIONS)
$(VERSIONS):
	cython -a $(SRC_DIR)/$@.pyx
	python $(SRC_DIR)/setup_$@.py build_ext -fi
	mv $@*.so $(LIB_DIR)
	mv $(SRC_DIR)/$@.c $(SRC_DIR)/$@.cpp $(SRC_DIR)/$@.html $(GEN_DIR)	

.PHONY: clean
clean:
	@rm -r $(GEN_DIR)/*.c $(GEN_DIR)/*.cpp $(GEN_DIR)/*.html ./build/ $(SRC_DIR)/__pycache__/
