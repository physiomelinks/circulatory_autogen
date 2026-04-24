# Compiler
CXX = g++
#CXX = mpic++
# Compiler flags (add -g for debug, or -O2/-O3 for optimization, or -Wall for enabling warning messages)
CXXFLAGS = -Wall -O2 -std=c++17 $(INC_DIR)

# SUNDIALS paths from Spack
# SUNDIALS_PREFIX := $(shell spack location -i sundials)
SUNDIALS_PREFIX := $(shell spack location -i sundials~mpi)
INC_DIR = -I$(SUNDIALS_PREFIX)/include
LIB_DIR = -L$(SUNDIALS_PREFIX)/lib

# Linker flags
LDFLAGS  = $(LIB_DIR)
# Linker libraries
LDLIBS = -lsundials_cvodes \
	 -lsundials_sunmatrixdense \
         -lsundials_sunlinsoldense \
         -lsundials_nvecserial \
         -lsundials_core \
         -lm

# Source files
SRCS = main0d.cpp model0d.cc
# Object files (auto-generated from SRCS)
OBJS := $(SRCS:.cpp=.o) # OBJS = $(SRCS:.cpp=.o)
OBJS := $(OBJS:.cc=.o)
# Output/target binary
TARGET = main0d

all: $(TARGET)

# Link target	
$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS) $(LDLIBS)

# Compile source files
%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@
%.o: %.cc
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Clean
clean:
	rm -f $(OBJS) $(TARGET)

