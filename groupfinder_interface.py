##### Python module to interface with C++ GroupFinder code #####
# TODO: include options for completeness testing, saving mass functions
# TODO: add option to run in parallel
# TODO: add in distance/velocity metric based inputs for simulation data 

import subprocess
import os
import h5py
import json
from dataclasses import dataclass
import numpy as np

@dataclass
class SimulationData:
    """Container for simulation galaxy data."""
    positions: np.ndarray  # array of shape (n_galaxies, 3) in Mpc in cartesian box coordinates
    velocities: np.ndarray  # array of shape (n_galaxies, 3) in km/s in cartesian box coordinates
    masses: np.ndarray  # array of shape (n_galaxies,) in log10(M_sun)
    ids: np.ndarray  # array of shape (n_galaxies,) unique identifiers
    ref_positions: np.ndarray  # Shape: (n_ref, 3) in Mpc in cartesian box coordinates
    ref_velocities: np.ndarray  # Shape: (n_ref, 3) in km/s in cartesian box coordinates
    ref_ids: np.ndarray = None  # optional, shape: (n_ref,) unique identifiers for reference points
    group_indices: np.ndarray = None  # optional, shape: (n_galaxies,) group indices if pre-grouped
    
    def __post_init__(self):
        """Validate that all arrays have consistent lengths and expected shapes."""
        n = len(self.positions)
        if not (len(self.velocities) == len(self.positions) == len(self.masses) == len(self.ids) == n):
            raise ValueError("All data arrays must have the same length")
        if self.positions.shape != (n, 3):
            raise ValueError("Positions array must have shape (n_galaxies, 3)")
        if self.velocities.shape != (n, 3):
            raise ValueError("Velocities array must have shape (n_galaxies, 3)")
        if not (len(self.ref_velocities) == len(self.ref_positions)):
            raise ValueError("All reference point arrays must have the same length")
        if self.ref_positions.shape != (3,) and self.ref_positions.shape[0] != 1:
            raise ValueError("Position array must have shape (3,) or (1, 3)")
        if self.ref_velocities.shape != (3,) and self.ref_velocities.shape[0] != 1:
            raise ValueError("Velocity array must have shape (3,) or (1, 3)")
    
    def write_to_hdf5(self, filename: str):
        """Write the simulation data to an HDF5 file."""
        with h5py.File(filename, 'w') as f:
            f.create_dataset('positions', data=self.positions)
            f.create_dataset('velocities', data=self.velocities)
            f.create_dataset('masses', data=self.masses)
            f.create_dataset('ids', data=self.ids)
            # Reshape ref positions/velocities to 2D (1, 3) for C++ compatibility
            f.create_dataset('ref_positions', data=self.ref_positions.reshape(1, 3))
            f.create_dataset('ref_velocities', data=self.ref_velocities.reshape(1, 3))
            if self.group_indices is not None:
                f.create_dataset('group_indices', data=self.group_indices)

@dataclass
class ObservationalData:
    """
    Container for observational galaxy data.
    Note: assumes user provides a physical distance (e.g., in Mpc) rather than redshift/velocity
    """
    masses: np.ndarray  # Shape: (n_observed,) in log10(M_sun)
    positions: np.ndarray  # Shape: (n_observed, 3) in spherical coordinates (distance, Dec, RA)
    velocities: np.ndarray  # Shape: (n_observed,) for line-of-sight/heliocentric velocities
    ids: np.ndarray  # Shape: (n_observed,) unique identifiers
    group_indices: np.ndarray = None  # optional, shape: (n_observed,) group indices if pre-grouped

    def __post_init__(self):
        """Validate that all arrays have consistent lengths and expected shapes."""
        n = len(self.positions)
        if not (len(self.velocities) == len(self.masses) == len(self.positions) == len(self.ids) == n):
            raise ValueError("All data arrays must have the same length")
        if self.positions.shape[1] != 3:
            raise ValueError("Positions array must have shape (n_observed, 3) corresponding to spherical coordinates (distance, Dec, RA)")
        if self.velocities.ndim != 1:
            raise ValueError("Velocities array must be one-dimensional corresponding to heliocentric velocities")
        
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
            f.create_dataset('masses', data=self.masses)
            f.create_dataset('positions', data=self.positions)
            f.create_dataset('velocities', data=self.velocities)
            f.create_dataset('ids', data=self.ids)
            if self.group_indices is not None:
                f.create_dataset('group_indices', data=self.group_indices)

class GroupFinderInterface:
    def __init__(self):
        self.dim = 3 # either 3D based (for 2 RA/dec + velocity) or 6D based (for x/y/z + vx/vy/vz)
        # relevant for simulation data only
        self.R_h_group_val = 1.0
        self.V_vir_group_val = 3.0
        self.R_h_iso_val = 2.0
        self.V_vir_iso_val = 3.0
        self.vel_cut_val = True
        self.tree_search_val = False
        self.sat_reclass_val = True
        self.iso_reclass_val = True
        self.contrast_val = True
        self.box_size = 50.0  # Example box size in Mpc; not required for observational data
        self.R_max = 50.0 * np.sqrt(3) / 2.0  # Example max distance for search
        self.periodic = True  # Periodic boundary conditions
        self.B_scaling = 1.0  # Scaling factor for B in density-contrast classification
        self.h = 0.6774  # Hubble parameter for unit conversions
    def config(self, filename: str, manual=False, obs=False):
        # parse args 
        if manual is True:
            print("GroupFinder Configuration")
            self.R_h_group_val = float(input(f'R_h_group (satellites grouped if d < R_h_group * R_h) [current={self.R_h_group_val}]: ') or self.R_h_group_val)
            self.V_vir_group_val = float(input(f'V_vir_group (satellites grouped if v < V_vir_group * V_vir) [current={self.V_vir_group_val}]: ') or self.V_vir_group_val)
            self.R_h_iso_val = float(input(f'R_h_iso (galaxy is isolated if d > R_h_iso * R_h) [current={self.R_h_iso_val}]: ') or self.R_h_iso_val)
            self.V_vir_iso_val = float(input(f'V_vir_iso (galaxy is isolated if v > V_vir_iso * V_vir) [current={self.V_vir_iso_val}]: ') or self.V_vir_iso_val)
            self.sat_reclass_val = input(f'Satellite reclassification (True/False) [current={self.sat_reclass_val}]: ').lower() in ('true', '1', 'yes') if input else self.sat_reclass_val
            self.iso_reclass_val = input(f'Isolated galaxy reclassification (True/False) [current={self.iso_reclass_val}]: ').lower() in ('true', '1', 'yes') if input else self.iso_reclass_val
            self.contrast_val = input(f'Density-contrast-based classification (True/False) [current={self.contrast_val}]: ').lower() in ('true', '1', 'yes') if input else self.contrast_val
            if self.contrast_val is False:
                self.vel_cut_val = input(f'Use velocity in classification (True/False) [current={self.vel_cut_val}]: ').lower() in ('true', '1', 'yes') if input else self.vel_cut_val
            else:
                pass  # velocity cut not applicable for density-contrast classification
            self.R_max = float(input(f'Max search radius R_max [current={self.R_max}]: ') or self.R_max)
            self.B_scaling = float(input(f'B scaling factor for density-contrast classification [current={self.B_scaling}]: ') or self.B_scaling)
            self.h = float(input(f'Hubble parameter h for unit conversions [current={self.h}]: ') or self.h)
            if obs is False:
                self.dim = int(input(f'Data dimension (3 for RA/Dec/velocity, 6 for x/y/z/vx/vy/vz) [current={self.dim}]: ') or self.dim)
                self.box_size = float(input(f'Box size [current={self.box_size}]: ') or self.box_size)
                self.periodic = input(f'Periodic boundary conditions (True/False) [current={self.periodic}]: ').lower() in ('true', '1', 'yes') if input else self.periodic
                if self.dim == 6:
                    self.tree_search_val = input(f'Use tree search: recommended for use only with 6D classification (True/False) [current={self.tree_search_val}]: ').lower() in ('true', '1', 'yes') if input else self.tree_search_val
            else:
                pass # observational data does not require box size
                # assumes non-periodic by default, and
                # tree search not applicable for RA/Dec/velocity data
            
        else:
            pass  # use default values

        # write values to json file
        with open(filename, 'w') as f:
            json.dump({
                "obs": obs,
                "R_h_group": self.R_h_group_val,
                "V_vir_group": self.V_vir_group_val,
                "R_h_iso": self.R_h_iso_val,
                "V_vir_iso": self.V_vir_iso_val,
                "vel_cut": self.vel_cut_val if not self.contrast_val else True,
                "tree_search": self.tree_search_val if self.dim == 6 and not obs else False,
                "sat_reclass": self.sat_reclass_val,
                "iso_reclass": self.iso_reclass_val,
                "contrast": self.contrast_val,
                "box_size": self.box_size if not obs else None,
                "R_max": self.R_max,
                "periodic": self.periodic if not obs else False,
                "B_scaling": self.B_scaling,
                "h": self.h,
                "dim": self.dim if not obs else 3
            }, f, indent=4)

    def calculate_B(self, sim_data: SimulationData, obs_data: ObservationalData, mass_limit: float, R_sim: float, R_obs: float) -> None:
        """
        Calculate the B parameter for density-contrast classification.
        Inputs:
            sim_data: SimulationData object containing simulation galaxy data.
            obs_data: ObservationalData object containing observational galaxy data.
            mass_limit: Stellar mass limit in log10(M_sun).
            R_sim: Cubic box size of simulation volume [Mpc] (or subset considered in GroupFinder).
            R_obs: Radius of volume-limited observational data [Mpc] (or subset considered in GroupFinder).
        """
        # Count number of simulated galaxies above mass limit
        sim_mask = sim_data.masses >= mass_limit
        N_sim = np.sum(sim_mask.reshape(-1))
        sim_vol = R_sim**3 # assumed cubic
        n_sim = N_sim / sim_vol

        # Count number of observed galaxies above mass limit within radius R
        obs_mask = (obs_data.masses >= mass_limit) & (obs_data.positions[:,0] <= R_obs) # ensure within distance R_obs
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
    The main function to run the GroupFinder code.
    Inputs:
        target: 'obs' for observational data or 'sim' for simulation data.
        infile: input HDF5 file name(s) containing galaxy data.
        outfile: output HDF5 file name(s) to save group finder results.
        config_file: JSON configuration file name.
        build: Whether to build the GroupFinder executable before running.
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