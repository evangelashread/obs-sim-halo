from groupfinder_interface import GroupFinderInterface, SimulationData, run_groupfinder
from input import InterpolationData
from tests.gen_data import GroupFinderTest
import h5py
import numpy as np

interface = GroupFinderInterface()

# set up config
interface.R_h_group = 1.0
interface.V_vir_group = 3.0
interface.R_h_iso = 2.0
interface.V_vir_iso = 3.0
interface.contrast = True # apply the Yang05 algorithm
interface.dim = 3
interface.tree_search = True
interface.sat_reclass = True
interface.iso_reclass = True
interface.h = 0.6774
interface.omega_M = 0.3089
interface.periodic = True
interface.box_size = 50

# Generate simulation test data (or include your own)
test = GroupFinderTest(box_size=interface.box_size, h=interface.h, omega_M=interface.omega_M)
test.create_test_data(type="sim", outfile="input/data/sim_data.h5", n_groups=50, n_sats=500)

with h5py.File("input/data/sim_data.h5", "r") as f:
    sim_positions = f['positions'][:] # in physical comoving Cartesian box coords [Mpc]
    sim_velocities = f['velocities'][:] # in physical Cartesian box coords [km/s]
    sim_masses = f['masses'][:] # in physical log stellar masses [log10(M_sun)]
    sim_ids = f['ids'][:]
    sim_ref_positions = f['ref_positions'][:]
    sim_ref_velocities = f['ref_velocities'][:]

sim_data = SimulationData(positions=sim_positions,
                             velocities=sim_velocities,
                             masses=sim_masses,
                             ids=sim_ids,
                             ref_positions=sim_ref_positions,
                             ref_velocities=sim_ref_velocities)

sim_data.write_to_hdf5('input/data/sim_data.h5')
print("Loaded simulation data.")

interface.config("input/sim_config.json", obs=False)

# run halo concentration data generation
InterpolationData.generate_concentration_data(max_z = 0.02)

# run redshift-distance data generation
InterpolationData.generate_z_dist_data(max_z = 0.02)

# run SMHM data generation
InterpolationData.generate_smhm_inverse_data(max_z = 0.02)

run_groupfinder('sim', 'input/data/sim_data.h5', 'sim_gf_result.h5', 'input/sim_config.json')