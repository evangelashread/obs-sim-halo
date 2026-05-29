from groupfinder_interface import GroupFinderInterface, ObservationalData, SimulationData, run_groupfinder
from input import InterpolationData
from tests.gen_data import GroupFinderTest
import h5py
import numpy as np

interface = GroupFinderInterface()

# Set up config
interface.R_h_group_val = 1.0
interface.V_vir_group_val = 3.0
interface.R_h_iso_val = 2.0
interface.V_vir_iso_val = 3.0
interface.contrast_val = True
interface.sat_reclass_val = True
interface.iso_reclass_val = True
interface.R_max = 50.0
interface.h = 0.6774
interface.omega_M = 0.3089
interface.use_distance = True

# Generate observational test data
test = GroupFinderTest(box_size=interface.R_max, h=interface.h, omega_M=interface.omega_M)
test.create_test_data(type="obs", outfile="input/data/test_obs_data.h5")

with h5py.File("input/data/test_obs_data.h5", "r") as f:
    obs_positions = np.array(f['positions'][:]) # already in spherical coords (dist [Mpc], dec [deg], RA [deg])
    obs_velocities = np.array(f['velocities'][:]) # heliocentric velocities [km/s]
    obs_masses = np.array(f['masses'][:]) # stellar masses [log10(M_sun)]
    obs_ids = np.array(f['ids'][:])
    obs_group_indices = np.array(f['group_indices'][:])
obs_data = ObservationalData(positions=obs_positions,
                             velocities=obs_velocities,
                             masses=obs_masses,
                             ids=obs_ids,
                             group_indices=obs_group_indices)
obs_data.write_to_hdf5('input/data/obs_data.h5')
print("Loaded observational data.")

# Generate simulation test data for B parameter calculation
test_sim = GroupFinderTest(box_size=35/interface.h, h=interface.h, omega_M=interface.omega_M)
test_sim.create_test_data(type="sim", outfile="input/data/sim_data.h5", n_groups=4)

with h5py.File("input/data/sim_data.h5", "r") as f:
    sim_positions = np.array(f['positions'][:]) # in physical Cartesian box coords [Mpc]
    sim_velocities = np.array(f['velocities'][:]) # in physical Cartesian box coords [km/s]
    sim_masses = np.array(f['masses'][:]) # in physical stellar masses [log10(M_sun)]
    sim_ids = np.array(f['ids'][:])
    sim_ref_positions = np.array(f['ref_positions'][:])
    sim_ref_velocities = np.array(f['ref_velocities'][:])

sim_data = SimulationData(positions=sim_positions,
                        velocities=sim_velocities,
                        masses=sim_masses,
                        ids=sim_ids,
                        ref_positions=sim_ref_positions,
                        ref_velocities=sim_ref_velocities)
    
# Calculate B parameter for data set A
interface.calculate_B(sim_data=sim_data,
                      obs_data=obs_data,
                      mass_limit=6.0,
                      R_sim=35.0/interface.h,
                      R_obs=50.0)

interface.config("input/obs_config.json", obs=True)

# Run halo concentration data generation
InterpolationData.generate_concentration_data(max_z = 0.03)

# Run redshift-distance data generation
InterpolationData.generate_z_dist_data(max_z = 0.03)

run_groupfinder('obs', 'input/data/obs_data.h5', 'obs_gf_result.h5', 'input/obs_config.json')