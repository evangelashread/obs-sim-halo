CXX       := g++
CXXFLAGS  := -I/path/to/Aboria/src -I/path/to/include/directory -O3 -std=c++17 -Wall -Wextra -march=native -Wno-missing-field-initializers -Wno-deprecated-copy -Wno-unused-parameter -Wno-type-limits -Wno-implicit-fallthrough -Wno-sign-compare
# Edit the first two -I flags to point to the Aboria (/path/to/Aboria/src) and include directories (/path/to/include/directory) on your system
# Usually the include path is your home directory (~) or /usr/local/include

# HDF5 configuration
HDF5_CXXFLAGS = -I/usr/include/hdf5/serial
HDF5_LDFLAGS  = -L/usr/lib/x86_64-linux-gnu/hdf5/serial -lhdf5_cpp -lhdf5

# Source files
SOURCES := GroupFinderCore.cpp GroupFinderStatistics.cpp DataIO.cpp
BUILD_DIR := build
OBJECTS := $(addprefix $(BUILD_DIR)/, $(SOURCES:.cpp=.o))

# Executables
TARGETS := run_groupfinder_obs run_groupfinder_sim

.PHONY: all clean obs sim

all:
	@echo "Please specify a target: obs or sim"
	@echo "  make obs          - Build run_groupfinder_obs"
	@echo "  make sim          - Build run_groupfinder_sim"

obs: run_groupfinder_obs

sim: run_groupfinder_sim

# Create build directory
$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

# Executable build rules
run_groupfinder_obs: $(BUILD_DIR) $(BUILD_DIR)/run_groupfinder_obs.o $(OBJECTS)
	$(CXX) $(CXXFLAGS) $(HDF5_CXXFLAGS) -o $(BUILD_DIR)/$@ $(BUILD_DIR)/run_groupfinder_obs.o $(OBJECTS) $(HDF5_LDFLAGS)

run_groupfinder_sim: $(BUILD_DIR) $(BUILD_DIR)/run_groupfinder_sim.o $(OBJECTS)
	$(CXX) $(CXXFLAGS) $(HDF5_CXXFLAGS) -o $(BUILD_DIR)/$@ $(BUILD_DIR)/run_groupfinder_sim.o $(OBJECTS) $(HDF5_LDFLAGS)

# Object file build rules
$(BUILD_DIR)/%.o: %.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(HDF5_CXXFLAGS) -c $< -o $@

# Dependencies
$(BUILD_DIR)/run_groupfinder_obs.o: run_groupfinder_obs.cpp GroupFinderCore.hh DataIO.hh
$(BUILD_DIR)/run_groupfinder_sim.o: run_groupfinder_sim.cpp GroupFinderCore.hh DataIO.hh
$(BUILD_DIR)/GroupFinderCore.o: GroupFinderCore.cpp GroupFinderCore.hh
$(BUILD_DIR)/GroupFinderStatistics.o: GroupFinderStatistics.cpp GroupFinderStatistics.hh
$(BUILD_DIR)/DataIO.o: DataIO.cpp DataIO.hh

clean:
	rm -rf $(BUILD_DIR)
