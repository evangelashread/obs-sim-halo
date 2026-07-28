from groupfinder_interface import GroupFinderInterface, ObservationalData, SimulationData, run_groupfinder
from input import InterpolationData
from tests.gen_data import GroupFinderTest, check_results, plot_groups
import h5py
import numpy as np
import astropy.cosmology
from astropy.cosmology import z_at_value
from astropy import units as u

cosmo = astropy.cosmology.Planck15

interface = GroupFinderInterface()

###########################################################
# Specifically used for mock lightcone data. Note that this 
# should produce the same results as if we use dim=3 in sim 
# mode with contrast=True, but in terms of implementation,
# this is ideal for data that is already in the format of a 
# mock galaxy lightcone.
###########################################################

# Set up config
interface.h = 0.6774
interface.omega_M = 0.3089
interface.box_size = 500/interface.h

# Generate simulation test data 
test_sim = GroupFinderTest(box_size=interface.box_size, h=interface.h, omega_M=interface.omega_M)
test_sim.create_test_data(type="sim", outfile="input/data/sim_data.h5", n_groups=50, n_sats=500, origin=True)

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
    
# Now transform from (x, y, z, v_x, v_y, v_z) -> (z, RA, Dec)

# Helper functions

def wrap(delta, box_size):
    return delta - box_size * np.round(delta / box_size)

def cartesian_to_celestial_coordinates(cartesian_coords: np.ndarray, 
                                       origin: np.ndarray = np.array([0.,0.,0]), 
                                       box_size=None) -> tuple:
    """
    Converts from Cartesian to celestial coordinates.
    Inputs:
        cartesian_coords: A numpy array of positions [x, y, z].
        origin: Optional. Box coordinates of the origin.

    Output:
        r, ra, dec: Converted celestial coordinates (r, phi, theta).

        r has bounds [0, inf) in same unit as input.
        ra has bounds[0, 2pi) rad
        dec has bounds [-pi/2, pi/2] deg
    """
    rel_coords = cartesian_coords - origin

    if box_size is not None:
        rel_coords = wrap(rel_coords, box_size)

    r = np.linalg.norm(rel_coords, axis=1)

    with np.errstate(divide='ignore', invalid='ignore'):
        cos_theta = np.clip(np.where(r > 0., rel_coords[:, 2] / r, 0.0), -1.0, 1.0)
        dec = np.pi / 2 - np.arccos(cos_theta)
        ra = np.mod(np.arctan2(rel_coords[:, 1], rel_coords[:, 0]), 2 * np.pi)

        pole = (np.abs(dec - np.pi / 2) < 1e-12) | (np.abs(dec + np.pi / 2) < 1e-12)
        ra = np.where(pole, 0.0, ra).reshape(-1)

        dec = np.where(r > 0., dec, 0.0).reshape(-1)
        ra = np.where(r > 0., ra, 0.0).reshape(-1)

    return r, ra, dec

def total_redshift(cart_positions: np.ndarray, 
                   cart_velocities: np.ndarray, 
                   origin: np.ndarray = np.array([0.,0.,0.]), 
                   v_at_origin: np.ndarray = np.array([0.,0.,0.]), 
                   box_size=None) -> np.ndarray:
    """
    Inputs:
        cel_positions: Numpy array of [x, y, z] of native box coordinates.
        cart_velocities: Numpy array of [v_x, v_y, v_z] of native box coordinate velocities.
        origin: Optional. Box coordinates of the origin.
        v_at_origin: Optional. The velocity of the origin.

    Output:
        The total redshift (cosmological and peculiar combined).
    """
    pos_rel = cart_positions - origin
    if box_size is not None:
        pos_rel = wrap(pos_rel, box_size)
    r = np.linalg.norm(pos_rel, axis=1)

    v_rel = cart_velocities - v_at_origin
    n_hat = pos_rel / r[:, None]
    v_los = np.sum(v_rel * n_hat, axis=1)

    z_cosmo = z_at_value(cosmo.comoving_distance, r * u.Mpc).value
    z_pec = v_los / 299792.458 # non-relativistic
    z_obs = (1 + z_cosmo) * (1 + z_pec) - 1

    return z_obs

z_total = total_redshift(sim_positions, sim_velocities, box_size=interface.box_size)
_, ra, dec = cartesian_to_celestial_coordinates(sim_positions, box_size=interface.box_size)

pos_all = np.vstack([z_total, dec, ra]).T

# Now we have a mock galaxy lightcone
obs_data = ObservationalData(positions=pos_all,
                             masses=sim_masses, # unchanged
                             ids=sim_ids) # unchages

obs_data.write_to_hdf5("input/data/lightcone_data.h5")

interface.R_h_group = 1.0
interface.V_vir_group = 3.0
interface.R_h_iso = 2.0
interface.V_vir_iso = 3.0
interface.contrast = True
interface.sat_reclass = True
interface.iso_reclass = True
interface.use_distance = False
interface.tree_search = True
interface.use_nanoflann = True
interface.config("input/lightcone_config.json", obs=True)

# Run halo concentration data generation
InterpolationData.generate_concentration_data(max_z = 0.03)

# Run redshift-distance data generation
InterpolationData.generate_z_dist_data(max_z = 0.03)

# Run stellar-mass-halo-mass relation data generation
InterpolationData.generate_smhm_inverse_data(max_z = 0.03)

run_groupfinder('obs', 'input/data/lightcone_data.h5', 'lightcone_gf_result.h5', 'input/lightcone_config.json')

check_results('input/data/sim_data.h5', 'lightcone_gf_result.h5')
#plot_groups('input/data/sim_data.h5', 'lightcone_gf_result.h5', 'sim')