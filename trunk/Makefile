# Binary file
BIN			= potts4-glass

# Flags
CUDA_INSTALL_PATH	?= /usr
#CUDA_INSTALL_PATH	= /usr/local/cuda
CUDASDK_INSTALL_PATH	= /opt/cudasdk
CUDA_LIB_PATH		= $(CUDA_INSTALL_PATH)/lib64
CUDASDK_LIB_PATH	= $(CUDASDK_INSTALL_PATH)/C/lib
INCLUDES += -I. -I$(CUDA_INSTALL_PATH)/include -I$(CUDASDK_INSTALL_PATH)/C/common/inc
LIBS			= -L$(CUDA_LIB_PATH) -L$(CUDASDK_LIB_PATH)
CFLAGS			= -O3 -Wall
LDFLAGS			= -lrt -lm -lcudart #-lcutil
NVCC_FLAGS		= -O3 -arch=sm_20 --ptxas-options=-v --use_fast_math

# Default Values
Q			= 5
L			= 4096
SAMPLES			= 1
TEMPERATURE		= 0.175
TMAX			= 20000
TWAIT			= 10000
NPOINTS			= 100
DELTA_T			= 1
OFILE			= energy[$(Q),$(L),$(SAMPLES),$(TEMPERATURE),$(TMAX),$(TWAIT),$(DELTA_T)].dat

# Simulation Parameters
PARAMETERS		= -DQ=$(Q) -DL=$(L) -DSAMPLES=$(SAMPLES) \
			-DTMAX=$(TMAX) -DTWAIT=$(TWAIT) -DTEMPERATURE=$(TEMPERATURE) \
			-DNPOINTS=$(NPOINTS) -DDELTA_T=$(DELTA_T)

# Compilers
NVCC			= $(CUDA_INSTALL_PATH)/bin/nvcc
GCC			= gcc
LINKER			= gcc

# Files
C_SOURCES		=
CU_SOURCES		= $(BIN).cu
HEADERS			=
C_OBJS			= $(patsubst %.c, %.o, $(C_SOURCES))
CU_OBJS			= $(patsubst %.cu, %.o, $(CU_SOURCES))


# Rules
$(BIN): clean $(C_OBJS) $(CU_OBJS) $(HEADERS)
	$(LINKER) -o $(BIN) $(CU_OBJS) $(C_OBJS) $(LDFLAGS) $(INCLUDES) $(LIBS)
 
$(C_OBJS): $(C_SOURCES) $(HEADERS)
	$(CC) -c $(C_SOURCES) $(CFLAGS) $(INCLUDES)
 
$(CU_OBJS): $(CU_SOURCES) $(HEADERS)
	$(NVCC) $(NVCC_FLAGS) -c $(CU_SOURCES) $(PARAMETERS) $(INCLUDES)
 
run: $(BIN)
	LD_LIBRARY_PATH=$(CUDA_LIB_PATH) ./$(BIN) > $(OFILE) &   
 
clean:
	rm -f $(BIN) *.o *.linkinfo cuda_profile_?.log
