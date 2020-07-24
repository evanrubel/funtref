# Detect Operating System
ifeq ($(OS),Windows_NT)
    detected_OS := Windows
else
    detected_OS := $(shell sh -c 'uname -s 2>/dev/null || echo not')
endif

# TODO: Set Windows Specific Environment Variables
ifeq ($(detected_OS),Windows)
    echo "Installation on Windows not currently supported."
endif

# Set MacOS Specific Environment Variables
ifeq ($(detected_OS),Darwin)
    EXT=.dylib
    LDFLAGS = -dynamiclib
endif

# Set Linux Specific Environment Variables
ifeq ($(detected_OS),Linux)
    EXT=.so
    LDFLAGS = -shared
endif

# If undefined (ie. not using conda-build), set PREFIX to active env
PREFIX ?= $(CONDA_PREFIX)

# Project Structure Dependent Variables
PROXTV = $(shell pwd)/external/proxtv
LIBPROXTV = $(PREFIX)/lib/libproxtv$(EXT)

GLMGEN = $(shell pwd)/external/glmgen
LIBGLMGEN = $(PREFIX)/lib/libglmgen$(EXT)

LIBTREFIDE = $(PREFIX)/lib/libtrefide$(EXT)

LDLIBS = -lzmq -lproxtv -lglmgen -lmkl_intel_lp64 -lmkl_core -lm -lmkl_intel_thread -liomp5

SRCS = $(wildcard src/*.cpp)
OBJS = $(patsubst %.cpp,%.o,$(SRCS))

INCLUDES = -I$(GLMGEN)/include -I$(PROXTV) -I$(PREFIX)/include
LDFLAGS += -L$(PREFIX)/lib

WARNINGS := -Wall -Wextra -pedantic -Weffc++ -Wshadow -Wpointer-arith \
            -Wcast-align -Wwrite-strings -Wmissing-declarations \
            -Wredundant-decls -Winline -Wno-long-long -Wconversion \
            -Wduplicated-cond -Wduplicated-branches -Wlogical-op -Wrestrict \
            -Wnull-dereference -Wold-style-cast -Wuseless-cast \
            -Wdouble-promotion -Wformat=2

CXXFLAGS := $(WARNINGS) -O3

# Compiler Dependent Environment Variables
CXX ?= g++

ifeq ($(CXX), icpc)
    CXXFLAGS += -mkl=sequential -qopenmp -fPIC $(INCLUDES) $(LDFLAGS) -D NOMATLAB=1
else
    CXXFLAGS += -fopenmp -fPIC $(INCLUDES) $(LDFLAGS) -D NOMATLAB=1
endif

# Recipes
.PHONY: all
all: clean $(LIBTREFIDE) $(LIBGLMGEN) $(LIBPROXTV)

$(LIBTREFIDE): $(OBJS) $(LIBGLMGEN) $(LIBPROXTV)
	$(CXX) $(CXXFLAGS) -o $@ $(OBJS) $(LDLIBS)

$(LIBPROXTV):
	$(MAKE) -C $(PROXTV);

$(LIBGLMGEN):
	$(MAKE) -C $(GLMGEN);

.PHONY: clean
clean:
	rm -f $(LIBTREFIDE) $(OBJS)
	$(MAKE) clean -C $(PROXTV);
	$(MAKE) clean -C $(GLMGEN);
