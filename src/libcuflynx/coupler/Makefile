# Compiler
CXX = g++
#CXX = mpic++
# Compiler flags (add -g for debug, or -O2/-O3 for optimization, or -Wall for enabling warning messages)
CXXFLAGS = -Wall -O2 -std=c++17 $(INC_DIR)
# CXXFLAGS = $(INC_DIR)

# nlohmann-json paths from Spack
JSON_PREFIX := $(shell spack location -i nlohmann-json)
INC_DIR = -I$(JSON_PREFIX)/include
# LIB_DIR = -L$(JSON_PREFIX)/lib

PIPE_DIR = /tmp
# PIPE_DIR = ./tmp

# # Linker flags
# LDFLAGS  = $(LIB_DIR)
# # Linker libraries
# LDLIBS = -lsundials_cvodes \
# 	 -lsundials_sunmatrixdense \
#          -lsundials_sunlinsoldense \
#          -lsundials_nvecserial \
#          -lsundials_core \
#          -lm

# Source files
SRCS = coupler.cpp coupler_auxFun.cpp
# Object files (auto-generated from SRCS)
OBJS := $(SRCS:.cpp=.o) # OBJS = $(SRCS:.cpp=.o)
# OBJS := $(OBJS:.cc=.o)
# Target binary/executable
TARGET = coupler

all: $(TARGET)

# Link target	
# $(TARGET): $(OBJS)
# 	$(CXX) $(CXXFLAGS) -o $@ $^
$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $(OBJS)

# Compile source files
%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@
# %.o: %.cc
# 	$(CXX) $(CXXFLAGS) -c $< -o $@

# Clean
clean:
	rm -f $(OBJS) $(TARGET)

clean_pipe:
	rm -f $(PIPE_DIR)/zero_to_*
	rm -f $(PIPE_DIR)/one_to_*
	rm -f $(PIPE_DIR)/parent_to_*

