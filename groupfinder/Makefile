CXX := g++

# Replace with path to your Aboria/src directory
ABORIA_DIR := /path/to/Aboria/src
# General include path to other header-only dependencies; currently confgured for macOS
INC_DIR := /opt/homebrew/include

# You'll possibly have to edit these sets of flags too; currently configured for macOS
HDF5_CXXFLAGS := -I/opt/homebrew/opt/hdf5/include
HDF5_LDFLAGS := -L/opt/homebrew/opt/hdf5/lib -lhdf5_cpp -lhdf5
OMP_CXXFLAGS := -I/opt/homebrew/opt/libomp/include -Xpreprocessor -fopenmp
OMP_LDFLAGS := -L/opt/homebrew/opt/libomp/lib -lomp

# Nothing here and below should ever need to be changed
CXXFLAGS := -I$(ABORIA_DIR) -I$(INC_DIR) $(OMP_CXXFLAGS) -O2 -std=c++17 -march=armv8.5-a -Wall -Wno-missing-field-initializers -Wno-deprecated-copy -Wno-unused-parameter -Wno-type-limits -Wno-implicit-fallthrough -Wno-sign-compare

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
	$(CXX) $(CXXFLAGS) $(HDF5_CXXFLAGS) -o $(BUILD_DIR)/$@ $(BUILD_DIR)/run_groupfinder_obs.o $(OBJECTS) $(HDF5_LDFLAGS) ${OMP_LDFLAGS}

run_groupfinder_sim: $(BUILD_DIR) $(BUILD_DIR)/run_groupfinder_sim.o $(OBJECTS)
	$(CXX) $(CXXFLAGS) $(HDF5_CXXFLAGS) -o $(BUILD_DIR)/$@ $(BUILD_DIR)/run_groupfinder_sim.o $(OBJECTS) $(HDF5_LDFLAGS) ${OMP_LDFLAGS}

$(BUILD_DIR)/%.o: %.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(HDF5_CXXFLAGS) -c $< -o $@

$(BUILD_DIR)/run_groupfinder_obs.o: run_groupfinder_obs.cpp GroupFinderCore.hh DataIO.hh
$(BUILD_DIR)/run_groupfinder_sim.o: run_groupfinder_sim.cpp GroupFinderCore.hh DataIO.hh
$(BUILD_DIR)/GroupFinderCore.o: GroupFinderCore.cpp GroupFinderCore.hh
$(BUILD_DIR)/GroupFinderStatistics.o: GroupFinderStatistics.cpp GroupFinderStatistics.hh
$(BUILD_DIR)/DataIO.o: DataIO.cpp DataIO.hh

clean:
	rm -rf $(BUILD_DIR)
