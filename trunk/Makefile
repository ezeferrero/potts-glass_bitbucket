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
Q                       = 3
L			= 4096
SAMPLES			= 1
TEMP	                = 0.1
TRAN                    = 0
TMAX                    = 2000
WAITING			= 1000
DELTA_T                 = 1
NPOINTS			= 80
TC                      = $(shell echo "1 / l(1+sqrt($Q))" | bc -l)
TEMPERATURE             = $(TEMP)
OFILE			= dinamica[CUDA,$(Q),$(L),$(SAMPLES),$(TEMP),$(TRAN),$(TMAX),$(WAITING),$(DELTA_T)].dat

# Simulation Parameters
PARAMETERS		= -DQ=$(Q) -DL=$(L) -DSAMPLES=$(SAMPLES) \
			-DTRAN=$(TRAN) -DTMAX=$(TMAX) -DWAITING=$(WAITING) \
			-DTEMPERATURE=$(TEMPERATURE) -DDELTA_T=$(DELTA_T) -DNPOINTS=$(NPOINTS)

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
