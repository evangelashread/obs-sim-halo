CXX := g++
# Edit the first two -I flags to point to the Aboria (/path/to/Aboria/src) and include directories (/path/to/include/directory) on your system
# Usually the include path is your home directory (~) or /usr/local/include
# May also need paths for OpenMP and HDF5
CXXFLAGS := -I/path/to/Aboria/src -I/path/to/include -I/path/to/libomp/include -I/path/to/hdf5/include -O2 -std=c++17 -Wall -Wextra -march=armv8.5-a -Wno-missing-field-initializers -Wno-deprecated-copy -Wno-unused-parameter -Wno-type-limits -Wno-implicit-fallthrough -Wno-sign-compare

# You'll possibly have to edit these two flags too; currently configured for macOS
HDF5_CXXFLAGS = -I/opt/homebrew/opt/hdf5/include
HDF5_LDFLAGS = -L/opt/homebrew/opt/hdf5/lib -lhdf5_cpp -lhdf5

SOURCES := GroupFinderCore.cpp GroupFinderStatistics.cpp DataIO.cpp
BUILD_DIR := build
OBJECTS := $(addprefix $(BUILD_DIR)/, $(SOURCES:.cpp=.o))

TARGETS := run_groupfinder_obs run_groupfinder_sim

.PHONY: all clean obs sim

all:
	@echo "Please specify a target: obs or sim"
	@echo "  make obs          - Build run_groupfinder_obs"
	@echo "  make sim          - Build run_groupfinder_sim"

obs: run_groupfinder_obs

sim: run_groupfinder_sim

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

run_groupfinder_obs: $(BUILD_DIR) $(BUILD_DIR)/run_groupfinder_obs.o $(OBJECTS)
	$(CXX) $(CXXFLAGS) $(HDF5_CXXFLAGS) -o $(BUILD_DIR)/$@ $(BUILD_DIR)/run_groupfinder_obs.o $(OBJECTS) $(HDF5_LDFLAGS)

run_groupfinder_sim: $(BUILD_DIR) $(BUILD_DIR)/run_groupfinder_sim.o $(OBJECTS)
	$(CXX) $(CXXFLAGS) $(HDF5_CXXFLAGS) -o $(BUILD_DIR)/$@ $(BUILD_DIR)/run_groupfinder_sim.o $(OBJECTS) $(HDF5_LDFLAGS)

$(BUILD_DIR)/%.o: %.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(HDF5_CXXFLAGS) -c $< -o $@

$(BUILD_DIR)/run_groupfinder_obs.o: run_groupfinder_obs.cpp GroupFinderCore.hh DataIO.hh
$(BUILD_DIR)/run_groupfinder_sim.o: run_groupfinder_sim.cpp GroupFinderCore.hh DataIO.hh
$(BUILD_DIR)/GroupFinderCore.o: GroupFinderCore.cpp GroupFinderCore.hh
$(BUILD_DIR)/GroupFinderStatistics.o: GroupFinderStatistics.cpp GroupFinderStatistics.hh
$(BUILD_DIR)/DataIO.o: DataIO.cpp DataIO.hh

clean:
	rm -rf $(BUILD_DIR)
