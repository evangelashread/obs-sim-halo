##### Python module to interface with C++ Group Finder code #####

import subprocess
import os
import h5py
import json
from dataclasses import dataclass
import numpy as np

@dataclass
class SimulationData:
    """Container for simulation galaxy data."""
    positions: np.ndarray  # shape (n_galaxies, 3) in Mpc in comoving cartesian box coordinates
    velocities: np.ndarray  # shape (n_galaxies, 3) in km/s
    masses: np.ndarray  # shape (n_galaxies,) in log10(M_sun)
    ids: np.ndarray  # shape (n_galaxies,) unique ids
    ref_positions: np.ndarray = None  # Shape: (1, 3) in Mpc in comoving cartesian box coordinates
    ref_velocities: np.ndarray = None  # Shape: (1, 3) in km/s
    ref_ids: np.ndarray = None  # optional, shape: (n_ref,) ids for the reference point(s)
    group_indices: np.ndarray = None  # optional, shape: (n_galaxies,) grouped galaxy ids if pre-grouped
    
    @staticmethod
    def validate_arrays(masses, ids, positions, velocities):
        """Shape/length check for input data. Can be called on whole data arrays or chunked data."""
        masses = np.asarray(masses)
        ids = np.asarray(ids)
        positions = np.asarray(positions)
        velocities = np.asarray(velocities)

        n = len(positions)
        if not (len(velocities) == len(positions) == len(masses) == len(ids) == n):
            raise ValueError("All data arrays must have the same length")
        if positions.shape != (n, 3):
            raise ValueError("Positions array must have shape (n_galaxies, 3)")
        if velocities.shape != (n, 3):
            raise ValueError("Velocities array must have shape (n_galaxies, 3)")

    @staticmethod
    def validate_ref_point(ref_positions, ref_velocities):
        """Check data for the single reference point."""
        ref_positions = np.asarray(ref_positions)
        ref_velocities = np.asarray(ref_velocities)
        if len(ref_velocities) != len(ref_positions):
            raise ValueError("All reference point arrays must have the same length")
        if ref_positions.shape != (3,) and ref_positions.shape[0] != 1:
            raise ValueError("Position array must have shape (3,) or (1, 3)")
        if ref_velocities.shape != (3,) and ref_velocities.shape[0] != 1:
            raise ValueError("Velocity array must have shape (3,) or (1, 3)")

    def __post_init__(self):
        """Validate that all arrays have consistent lengths and shapes."""
        SimulationData.validate_arrays(self.masses, self.ids, self.positions, self.velocities)
        if self.ref_positions is None or self.ref_velocities is None:
            self.ref_positions = np.zeros((1, 3))
            self.ref_velocities = np.zeros((1, 3))
        SimulationData.validate_ref_point(self.ref_positions, self.ref_velocities)

    def write_to_hdf5(self, filename: str):
        """Write the simulation data to an HDF5 file."""
        with h5py.File(filename, 'w') as f:
            f.create_dataset('positions', data=self.positions, chunks=True)
            f.create_dataset('velocities', data=self.velocities, chunks=True)
            f.create_dataset('masses', data=self.masses, chunks=True)
            f.create_dataset('ids', data=self.ids, chunks=True)
            # Reshape ref positions/velocities to 2D (1, 3) for C++ compatibility
            f.create_dataset('ref_positions', data=self.ref_positions.reshape(1, 3), chunks=True)
            f.create_dataset('ref_velocities', data=self.ref_velocities.reshape(1, 3), chunks=True)
            if self.group_indices is not None:
                f.create_dataset('group_indices', data=self.group_indices, chunks=True)

@dataclass
class ObservationalData:
    """
    Container for observational galaxy data.
    
    Positions must be provided as (comoving distance, Dec, RA) OR (z, Dec, RA) in that order.
    If comoving distance is provided, then the user must also provide peculiar velocities.

    RA AND DEC MUST BE IN RADIANS.
    """
    masses: np.ndarray  # Shape: (n_observed,) in log10(M_sun)
    ids: np.ndarray  # Shape: (n_observed,) unique ids
    positions: np.ndarray  # Shape: (n_observed, 3) in spherical coordinates (distance, Dec, RA) OR (redshift, Dec, RA)
    velocities: np.ndarray = None  # Shape: (n_observed,) for line-of-sight/heliocentric velocities
    group_indices: np.ndarray = None  # optional, shape: (n_observed,) galaxy group ids if pre-grouped

    @staticmethod
    def validate_arrays(masses, ids, positions, velocities=None):
        """Shape/value check on input data. Can be called on whole data arrays or chunks."""
        masses = np.asarray(masses)
        ids = np.asarray(ids)
        positions = np.asarray(positions)
        n = len(positions)
        if not (len(masses) == len(positions) == len(ids) == n):
            raise ValueError("All data arrays must have the same length")
        if positions.ndim != 2 or positions.shape[1] != 3:
            raise ValueError("Positions array must have shape (n, 3) corresponding to (distance/z, Dec, RA)")

        dec = positions[:, 1]
        ra = positions[:, 2]
        tol = 1e-6
        if np.any((dec < -np.pi / 2 - tol) | (dec > np.pi / 2 + tol)):
            raise ValueError("Dec values out of bounds [-pi/2, pi/2]")
        if np.any((ra < -tol) | (ra >= 2 * np.pi + tol)):
            raise ValueError("RA values out of bounds [0, 2*pi)")

        if velocities is not None:
            velocities = np.asarray(velocities)
            if velocities.ndim != 1:
                raise ValueError("Velocities array must be one-dimensional corresponding to heliocentric velocities")
            if len(velocities) != n:
                raise ValueError("Velocities array must have the same length as positions and masses")

    def __post_init__(self):
        """Validate that all arrays have consistent lengths and expected shapes."""
        ObservationalData.validate_arrays(self.masses, self.ids, self.positions, self.velocities)
        if self.velocities is None:
            self.velocities = [0.]

    def prep_coords(self, RA: np.ndarray, Dec: np.ndarray, distances: np.ndarray):
        """
        Converts celestial coordinates (RA, Dec, distance) to spherical coordinates (r, theta, phi).
        Inputs:
            RA: Array of right ascension in degrees.
            Dec: Array of declination in degrees.
            distances: Array of distances in Mpc.
        Adjust as needed based on input units.
        """
        import astropy.units as u
        from astropy.coordinates import SkyCoord
        celestial_coords = SkyCoord(ra=RA*u.degree, dec=Dec*u.degree, distance=distances*u.Mpc, frame='icrs')
        sph = celestial_coords.spherical
        r = sph.distance
        phi = sph.lon.to(u.rad).value
        theta = sph.lat.to(u.rad).value
        self.positions = np.vstack((r, theta, phi)).T
        
    def write_to_hdf5(self, filename: str):
        """Write the observational data to an HDF5 file."""
        with h5py.File(filename, 'w') as f:
            f.create_dataset('masses', data=self.masses, chunks=True)
            f.create_dataset('positions', data=self.positions, chunks=True)
            f.create_dataset('ids', data=self.ids, chunks=True)
            if self.velocities is not None:
                f.create_dataset('velocities', data=self.velocities, chunks=True)
            if self.group_indices is not None:
                f.create_dataset('group_indices', data=self.group_indices, chunks=True)

class GroupFinderInterface:
    def __init__(self):
        self.dim = 3 # either 3D based (for 2 RA/dec + velocity/redshift/distance) or 6D based (for x/y/z + vx/vy/vz)
        # relevant for simulation data only
        self.R_h_group = 1.0
        self.V_vir_group = 3.0
        self.R_h_iso = 2.0
        self.V_vir_iso = 3.0
        self.vel_cut = True # classify based on peculiar velocity (not applicable for density-contrast-based classification)
        self.tree_search = False
        self.sat_reclass = True
        self.iso_reclass = True
        self.contrast = True
        self.box_size = 50.0  # Example box size in Mpc; not required for observational data
        self.R_max = 50.0 * np.sqrt(3) / 2.0  # Example max distance for search
        self.periodic = True  # Periodic boundary conditions
        self.B_scaling = 1.0  # Scaling factor for B in density-contrast classification
        self.h = 0.6774  # Hubble parameter
        self.omega_M = 0.3089 # Matter density at z=0
        self.use_distance = True
        self.chunk = False
        self.chunk_size = 1_000_000
        self.R_h_max_override = -1.0 # option to override the default behavior of calculating a maximum search radius from the precomputed halo properties for all potential halos
            # Optionally set R_h_max to a value >0 Mpc 
        self.use_nanoflann = False
    def config(self, filename: str, obs=False):
        # write values to json file
        with open(filename, 'w') as f:
            json.dump({
                "obs": obs,
                "R_h_group": self.R_h_group,
                "V_vir_group": self.V_vir_group,
                "R_h_iso": self.R_h_iso,
                "V_vir_iso": self.V_vir_iso,
                "vel_cut": self.vel_cut if self.use_distance and not self.contrast else True,
                "tree_search": self.tree_search,
                "sat_reclass": self.sat_reclass,
                "iso_reclass": self.iso_reclass,
                "contrast": self.contrast,
                "box_size": self.box_size if not obs else None,
                "R_max": self.R_max,
                "periodic": self.periodic if not obs else False,
                "B_scaling": self.B_scaling,
                "h": self.h,
                "omega_M": self.omega_M,
                "dim": self.dim if not obs else 3,
                "use_distance": self.use_distance if obs else True,
                "chunk": self.chunk,
                "chunk_size": self.chunk_size,
                "R_h_max_override": self.R_h_max_override,
                "use_nanoflann": self.use_nanoflann,
            }, f, indent=4)

    def calculate_B(self, sim_data: SimulationData, obs_data: ObservationalData, mass_limit: float, R_sim: float, R_obs: float, cubic: bool = True) -> None:
        """
        Calculate the B parameter for density-contrast classification.
        Inputs:
            sim_data: SimulationData object containing simulation galaxy data.
            obs_data: ObservationalData object containing observational galaxy data.
            mass_limit: Stellar mass limit in log10(M_sun).
            R_sim: Cubic box size of simulation volume [Mpc] (or subset used in the group finder).
            R_obs: Radius of volume-limited observational data [Mpc] (or subset used in the groupfinder).
            cubic: Whether the simulation volume is cubic (True) or spherical (False). 
                   Observation volume is assumed spherical.
        """
        # Count number of simulated galaxies above mass limit 
        # (presumably SimulationData is already limited to the volume that will be considered in the group finder)
        sim_mask = (sim_data.masses >= mass_limit) 
        N_sim = np.sum(sim_mask.reshape(-1))
        if cubic:
            sim_vol = R_sim**3 # assumed cubic
        else:
            sim_vol = (4/3)*np.pi*R_sim**3  # assumed spherical
        n_sim = N_sim / sim_vol
        
        if not self.use_distance:
            import astropy.cosmology
            astropy_cosmo = astropy.cosmology.Planck15
            # convert redshifts to comoving distance
            obs_dists = astropy_cosmo.comoving_distance(obs_data.positions[:,0]).value
            obs_mask = (obs_data.masses >= mass_limit) & (obs_dists <= R_obs) # ensure within distance R_obs
        else:
            obs_mask = (obs_data.masses >= mass_limit) & (obs_data.positions[:,0] <= R_obs) # ensure within distance R_obs
        # Count number of observed galaxies above mass limit within radius R
        N_obs = np.sum(obs_mask.reshape(-1))
        obs_vol = (4/3)*np.pi*R_obs**3  # assumed spherical
        n_obs = N_obs / obs_vol

        # Calculate B scaling factor
        if N_sim == 0 or N_obs == 0:
            raise ValueError("No simulated galaxies found above the mass limit within the specified radius.")
        
        self.B_scaling = n_obs / n_sim
        print(f"Calculated B scaling factor: {self.B_scaling}")

class GroupFinderRunner:
    def __init__(self):
        self.ROOT = os.path.dirname(os.path.abspath(__file__))
        self.MAKEFILE_DIR = os.path.join(self.ROOT, "groupfinder")
        self.EXE = None

    def build(self, target: str):
        subprocess.run(["make", "-f", "Makefile", "clean"], cwd=self.MAKEFILE_DIR, check=True)
        subprocess.run(["make", target], cwd=self.MAKEFILE_DIR, check=True)

    def run_command(self, exe, infile, outfile, config_file: str):
        # update C++ files to take in config parameters
        exe_path = os.path.join(self.MAKEFILE_DIR, "build", exe)
        if isinstance(infile, list):
            infile = ','.join(infile)
        elif isinstance(infile, str):
            pass
        else:
            raise TypeError("infile must be a string or list of strings.")
        if isinstance(outfile, list):
            outfile = ','.join(outfile)
        elif isinstance(outfile, str):
            pass
        else:
            raise TypeError("outfile must be a string or list of strings.")
        cmd = [exe_path, infile, outfile, config_file]
        subprocess.run(cmd, cwd=self.ROOT, check=True)

def run_groupfinder(target: str, infile, outfile, config_file: str, build: bool = False):
    """
    The main function to run the group finder code.
    Inputs:
        target: 'obs' for observational data or 'sim' for simulation data.
        infile: input HDF5 file name(s) containing galaxy data.
        outfile: output HDF5 file name(s) to save group finder results.
        config_file: JSON configuration file name.
        build: Whether to build the group finder executable before running.
    """
    gfrun = GroupFinderRunner()
    if not (isinstance(infile, str) or isinstance(infile, list)):
        raise TypeError("infile must be a string or list of strings.")
    if not (isinstance(outfile, str) or isinstance(outfile, list)):
        raise TypeError("outfile must be a string or list of strings.")
    if isinstance(infile, str) and isinstance(outfile, list):
        raise TypeError("infile and outfile must be of the same type.")
    if isinstance(infile, list) and isinstance(outfile, str):
        raise TypeError("infile and outfile must be of the same type.")
    if target == 'obs':
        prefix_src = "run_groupfinder_obs"
        if isinstance(infile, list) or isinstance(outfile, list):
            raise TypeError("infile and outfile must be strings for observational data.")
    elif target == 'sim':
        prefix_src = "run_groupfinder_sim"
    else:
        raise ValueError("Input parameter target must be 'obs' or 'sim'.")
    gfrun.EXE = prefix_src
    if build:
        gfrun.build(target)
    gfrun.run_command(gfrun.EXE, infile, outfile, config_file)