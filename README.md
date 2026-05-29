# ObsSimHalo

A high-performance, halo-based galaxy group finding algorithm implemented in C++ with a Python interface, designed for consistent comparisons between observational and simulation data.

Note: this repo is still being updated.

## Overview

ObsSimHalo offers two main approaches for identifying galaxy groups:

1. A simple halo-based group finder that uses abundance matching to assign galaxies to groups based on the mass of their halo, estimated from a stellar-to-halo-mass relation ([Behroozi et al. 2019](https://doi.org/10.1093/mnras/stz1182)).

2. An adaptation of the popular [Yang et al. 2005](https://doi.org/10.1111/j.1365-2966.2005.08560.x) algorithm that classifies galaxies using their number density contrast in redshift/velocity space. This method is better suited for like-for-like comparisons of observational and simulation data.

This algorithm has been tested with [TNG50 data](https://www.tng-project.org/data/) and observational data sourced from the 50 Mpc Galaxy Catalog ([Ohlson et al. 2024](https://github.com/davidohlson/50MGC)) and the [DESI Extragalactic Dwarf Galaxy Catalog](https://data.desi.lbl.gov/doc/releases/dr1/vac/extragalactic-dwarfs/).

In the adaptation of the Yang et al. (2005) algorithm (density-contrast-based classification), the estimated worst-case time complexity is O(N^2) when a brute force search is used. For N=10^6, this particular algorithm runs in ~30 min on an Intel i7 13th gen core (Ubuntu) if the cutoff radius for the brute search exceeds the radius of the sample volume. The 6D classification is accelerated with the use of k-d trees, achieving approximately O(N log N) complexity. For N=10^6, the 6D algorithm runs in ~5 min.

While initially designed for classification at z ~ 0, this can also handle survey or simulation data out to any redshift. Flat lambdaCDM is assumed.

### Observationally Consistent Classification

This group finder can classify groups in simulated catalogs in a way that mimics observations. To do this, it reduces six-dimensional simulation data (3D position + 3D velocity) to three dimensions (RA-like, Dec-like, and line-of-sight velocity/distance/redshift). A reference position (such as a Milky Way analogue) can be specified to anchor the coordinate system if different from (0,0,0).

When using the density-contrast method, it's recommended to scale the parameter "B" if the galaxy population is known to be incomplete. In the examples, "B" is set to the ratio of observed to simulated galaxy number densities at a given mass/luminosity limit.

## Dependencies

### C++ Dependencies

- **C++ compiler**: g++ with C++17 support
- **HDF5**: Version 1.8+ with C++ bindings
  - Ubuntu/Debian: `sudo apt-get install libhdf5-dev libhdf5-cpp-103`
  - macOS: `brew install hdf5`
- **Aboria**: Header-only C++ library for neighbor searching
  - Clone from: https://github.com/martinjrobins/Aboria
  - Set include path in Makefile to Aboria location
- **nlohmann/json**: Header-only JSON library (usually at `/usr/include/nlohmann/`)
  - Ubuntu/Debian: `sudo apt-get install nlohmann-json3-dev`
  - macOS: `brew install nlohmann-json`

### Python Dependencies

Install via pip:
```bash
pip install -r requirements.txt
```

Required packages:
- numpy
- h5py
- colossus
- astropy
- scipy
- pytest

## Installation

1. **Clone the repository, and install Python dependencies**:
   ```bash
   cd obs-sim-halo
   pip install -r requirements.txt
   ```

2. **Configure C++ compilation**:
   This doesn't have a CMakeLists file at the moment, so after installing the C++ dependencies listed above, edit the Makefile in the `groupfinder/` directory to set paths for:
   - Aboria headers (`-I/path/to/Aboria/src`)
   - Include path (`-I/path/to/include/directory`)
   - HDF5 installation (if not in standard locations)

3. **Build the C++ executables**:
   ```bash
   cd groupfinder
   make obs  # For observational data processing
   make sim  # For simulation data processing
   ```

4. **Run tests**
   ```
   pytest tests/test_groupfinder.py
   ```
   Run to ensure your installation is working.

## Usage

### Observational Data

```python
from groupfinder_interface import GroupFinderInterface, ObservationalData, run_groupfinder
from input import InterpolationData

# Initialize interface
interface = GroupFinderInterface()

# Configure parameters
interface.h = 0.7
interface.omega_M = 0.3
# ...

# Write to JSON
interface.config('input/obs_config.json', obs=True)

# Prepare data
obs_data = ObservationalData(
    positions=p_array,         # Spherical coordinates (Distance [Mpc], Dec [rad], RA [rad]) OR (Redshift, Dec [rad], RA [rad]). Dec range: [-pi/2, pi/2], RA range: [0, 2pi)
    velocities=v_array,      # Peculiar line-of-sight velocity [km/s] OR None if redshifts used
    masses=mass_array,     # Stellar masses [log10(M_sun)]
    ids=id_array          # Galaxy IDs
)
obs_data.write_to_hdf5('input/data/input_file.h5')

# Generate halo-mass-concentration data and redshift-distance data for group finder
# Note that you may have to adjust the cosmology manually in InterpolationData (this will be made configurable in a later update)
InterpolationData.generate_concentration_data()
InterpolationData.generate_z_dist_data()

# Run group finder
run_groupfinder('obs', 'input/data/input_file.h5', 'output_file.h5', 'input/obs_config.json')
```

### Simulation Data

```python
from groupfinder_interface import SimulationData, run_groupfinder

# Prepare simulation data
sim_data = SimulationData(
    positions=pos_array,      # [N, 3] Comoving cartesian box positions [Mpc]
    velocities=vel_array,     # [N, 3] Peculiar velocities [km/s]
    masses=mass_array,        # Stellar masses [log10(M_sun)]
    ids=id_array,            # Galaxy IDs
    ref_positions=ref_pos,   # Reference position (e.g., MW analogue); set to None if in observer frame
    ref_velocities=ref_vel   # Reference velocity; set to None if already in observer frame
)

# Configure and run
interface.dim = 3
# ...
interface.config('input/sim_config.json')
run_groupfinder('sim', input_file(s), output_file(s), 'input/sim_config.json')
```

### Configuration Parameters

- `R_h_group`: Satellites grouped if distance < R_h_group × R_h (default: 1.0)
- `V_vir_group`: Satellites grouped if velocity < V_vir_group × V_vir (default: 3.0)
- `R_h_iso`: Galaxy isolated if distance > R_h_iso × R_h (default: 2.0)
- `V_vir_iso`: Galaxy isolated if velocity > V_vir_iso × V_vir (default: 3.0)
- `h`: Dimensionless Hubble parameter, h = H0 / 100 km/s/Mpc (default: 0.6774)
- `omega_M`: Matter density parameter at z=0
- `sat_reclass`: Enable satellites to be reclassified after initial classification (default: true)
- `iso_reclass`: Enable isolated galaxies to be reclassified after initial classification (default: true)
- `vel_cut`: Include velocity in classification criteria, as opposed to positions only (default: true)
- `tree_search`: Use a kdtree for efficient searching. Currently only supports 3D Cartesian positions (default: false)
- `contrast`: Use density-contrast-based classification (default: true)
- `B_scaling`: For use with density-contrast-based classification
- `dim`: Data dimension. 3 for RA/Dec/line-of-sight velocity+distance, 6 for full 6D phase space
- `box_size`: Length of 3D simulation box. Not needed for observation config
- `R_max`: Maximum search radius for brute force nearest neighbors search
- `use_distance`: Relevant for observational mode. Whether to use comoving distance + peculiar velocity OR redshift only.

## Output

Results are saved to HDF5 files with the following contents:

- **`group_member_ids`**: Galaxy groups, each stored as a list of original galaxy IDs. Single-member lists represent isolated galaxies.
- **`central_ids`**: The central galaxy in each group (including isolated galaxies).
- **Statistics and group properties**: Derived quantities for analyzing the results (total number of groups, total numbers of isolated galaxies and galaxies in multi-member groups).

## Examples

See `example_obs.py` and `example_sim.py` for complete working examples with user-generated test data. 

## Contributions

Contributions are welcome — please open an issue or pull request.

## Citation

If you use this code in your research, please cite [Shread et al. 2026](https://doi.org/10.3847/1538-4357/ae644c) as well as [Yang et al. 2005](https://doi.org/10.1111/j.1365-2966.2005.08560.x) and [Behroozi et al. 2019](https://doi.org/10.1093/mnras/stz1182).