"""
Module to generate test data for GroupFinder unit tests.
The last satellite associated with each group is always an isolated galaxy.

Note: Future versions will include more tests, particularly for edge cases.
"""

import numpy as np
import h5py
from scipy.optimize import fsolve
from pathlib import Path
import os
import sys
import astropy.cosmology
from astropy.cosmology import z_at_value
from astropy import units as u
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
from groupfinder_interface import SimulationData, ObservationalData

cosmo = astropy.cosmology.Planck15

class GroupFinderTest:
    def __init__(self, box_size, h, omega_M):
        self.box_size = box_size
        self.h = h
        self.omega_M = omega_M
        self.G = 4.3009172706e-9 # Mpc (km/s)^2 Msun^{-1}
        self.mass = None
        self.p_crit_0 = (3 * (h*100)**2) / (8 * np.pi * (4.3009172706e-9))
        # Predefined parameters for satellite generation
        # In further versions, these should be made configurable
        self.R_h_group = 1.0
        self.R_h_iso = 2.0
        self.V_vir_group = 3.0
        self.V_vir_iso = 3.0
        self.behroozi_params = {
            "EPS_0": -1.435,
            "EPS_A": 1.831,
            "EPS_LOGA": 1.368,
            "EPS_Z": -0.217,
            "M_0": 12.035,
            "M_A": 4.556,
            "M_LOGA": 4.417,
            "M_Z": -0.731,
            "ALPHA_0": 1.963,
            "ALPHA_A": -2.316,
            "ALPHA_LOGA": -1.732,
            "ALPHA_Z": 0.178,
            "BETA_0": 0.482,
            "BETA_A": -0.841,
            "BETA_Z": -0.471,
            "DELTA_0": 0.411,
            "GAMMA_0": -1.034,
            "GAMMA_A": -3.100,
            "GAMMA_Z": -1.055,
        }
        
    def p_crit(self, z):
        return self.p_crit_0 * (self.omega_M * (1 + z)**3 + (1 - self.omega_M))

    def behroozi_SMHM(self, x, z, Mstar_log):
        """
        Behroozi et al. 2019 SMHM relation

        Adapted from velociraptor python code
        """
        params = self.behroozi_params
        a = 1.0 / (1.0 + z)
        a1 = a - 1.0
        lna = np.log(a)

        zparams = {}

        zparams["m_1"] = (
            params["M_0"]
            + a1 * params["M_A"]
            - lna * params["M_LOGA"]
            + z * params["M_Z"]
        )
        zparams["eps"] = (
            params["EPS_0"]
            + a1 * params["EPS_A"]
            - lna * params["EPS_LOGA"]
            + z * params["EPS_Z"]
        )
        zparams["alpha"] = (
            params["ALPHA_0"]
            + a1 * params["ALPHA_A"]
            - lna * params["ALPHA_LOGA"]
            + z * params["ALPHA_Z"]
        )
        zparams["beta"] = params["BETA_0"] + a1 * params["BETA_A"] + z * params["BETA_Z"]
        zparams["delta"] = params["DELTA_0"]
        zparams["gamma"] = 10 ** (
            params["GAMMA_0"] + a1 * params["GAMMA_A"] + z * params["GAMMA_Z"]
        )

        x2 = x / zparams["delta"]
        logmstar = (
            zparams["eps"] + zparams["m_1"]
            - np.log10(10 ** (-zparams["alpha"] * x) + 10 ** (-zparams["beta"] * x))
            + zparams["gamma"] * np.exp(-0.5 * (x2 * x2))
        )
        
        return logmstar - Mstar_log
    
    def halo_props(self, z, Mstar_log):
        sol = fsolve(self.behroozi_SMHM, x0=3.0, args=(z, Mstar_log))[0]
        a = 1.0 / (1.0 + z)
        a1 = a - 1.0
        lna = np.log(a)
        M1 = self.behroozi_params["M_0"] + self.behroozi_params["EPS_0"] + a1 * self.behroozi_params["EPS_A"] - lna * self.behroozi_params["EPS_LOGA"] + z * self.behroozi_params["EPS_Z"]
        M_h = 10 ** (sol + M1)
        R_h = (3 * M_h / (4 * np.pi * 200 * self.p_crit(z))) ** (1/3)
        v_vir = np.sqrt(self.G * M_h / R_h)
        return M_h, R_h, v_vir
    
    def cart_to_sph(self, vec):
        """
        Converts from Cartesian to spherical coordinates.
            
        Inputs:
            vec: A numpy array like [x, y, z] for a single point or array of all points.
        
        Output: 
            r, theta, phi: The spherical coordinates for the single point.
        """
        x, y, z = vec
        r = np.sqrt(x**2 + y**2 + z**2)
        with np.errstate(divide='ignore', invalid='ignore'):
            # Handle the cases where r = 0 or when phi is undefined, which occurs for theta = 0
            cos_theta = np.clip(np.where(r > 0., z / r, 1.0), -1.0, 1.0)
            theta = np.arccos(cos_theta)
            phi = np.mod(np.arctan2(y, x), 2*np.pi)

            # Define phi on the poles to avoid noise
            pole = (theta < 1e-12) | (np.abs(theta - np.pi) < 1e-12)
            phi = np.where(pole, 0.0, phi).reshape(-1)

            # At r = 0, take the angle to be (0., 0.)
            theta = np.where(r > 0., theta, 0.0).reshape(-1)
            phi = np.where(r > 0., phi, 0.0).reshape(-1)

        if len(theta.shape) == 1:
            return r, theta[0], phi[0]
        else:
            return r, theta, phi

    def generate_satellites(self, log10_stellar_c, center_pos, center_vlos, N, sim=False):
        if N == 0:
            return np.empty((0,3)), np.empty((0,)), np.empty((0,))
        z_c = z_at_value(cosmo.comoving_distance, np.linalg.norm(center_pos) * u.Mpc).value
        _, R_h, v_vir = self.halo_props(z_c, log10_stellar_c)
        
        # Generate N total satellites
        cos_t = np.random.uniform(-1., 1., N)
        theta = np.arccos(cos_t)
        phi = np.random.uniform(0., 2*np.pi, N)
        m = np.random.uniform(6.0, 8.5, N) # ensure satellite mass < central mass
        
        if N > 1:
            # Generate N-1 bound satellites + 1 isolated
            N_bound = N - 1
            r = np.random.uniform(0.1*(self.R_h_group*R_h), 0.6*(self.R_h_group*R_h), N_bound)
            if sim:
                v = np.zeros((N, 3))
                for i in range(N_bound):
                    # Ensure velocity vector is 3D
                    v_mag = np.random.uniform(0., 0.5 * self.V_vir_group * v_vir)
                    vx = v_mag * np.sin(theta[i]) * np.cos(phi[i])
                    vy = v_mag * np.sin(theta[i]) * np.sin(phi[i])
                    vz = v_mag * np.cos(theta[i])
                    v_i = np.array([vx, vy, vz])
                    v[i] = v_i + np.array([center_vlos[0], center_vlos[1], center_vlos[2]])
                # Add isolated satellite, always as the last one
                v_iso_mag = np.random.uniform(1.5*self.V_vir_group*v_vir, 2*self.V_vir_group*v_vir) * np.random.choice([-1, 1])
                vx_iso = v_iso_mag * np.sin(theta[-1]) * np.cos(phi[-1])
                vy_iso = v_iso_mag * np.sin(theta[-1]) * np.sin(phi[-1])
                vz_iso = v_iso_mag * np.cos(theta[-1])
                v_iso = np.array([vx_iso, vy_iso, vz_iso]) + np.array([center_vlos[0], center_vlos[1], center_vlos[2]])
                v[-1] = v_iso  # Assign to the LAST satellite only
            else:
                v = np.random.uniform(-0.5*self.V_vir_group*v_vir, 0.5*self.V_vir_group*v_vir, N_bound) + center_vlos
                # Add isolated satellite
                v_iso = np.random.uniform(1.5*self.V_vir_group*v_vir, 2*self.V_vir_group*v_vir) * np.random.choice([-1, 1]) + center_vlos
                v = np.append(v, v_iso)
            r_iso = np.random.uniform(1.5*self.R_h_iso*R_h, 2*self.R_h_iso*R_h)
            r = np.append(r, r_iso)
            
        else:
            # Just one satellite, make it bound
            r = np.random.uniform(0.1*self.R_h_group*R_h, 0.6*self.R_h_group*R_h, N)
            if sim:
                v_mag = np.random.uniform(0, 0.5*self.V_vir_group*v_vir)
                vx = v_mag * np.sin(theta[0]) * np.cos(phi[0])
                vy = v_mag * np.sin(theta[0]) * np.sin(phi[0])
                vz = v_mag * np.cos(theta[0])
                v = (np.array([vx, vy, vz]) + np.array([center_vlos[0], center_vlos[1], center_vlos[2]])).reshape(1, 3)
            else:
                v = np.random.uniform(-0.5*self.V_vir_group*v_vir, 0.5*self.V_vir_group*v_vir, N) + center_vlos
            
        rel = np.column_stack([r*np.sin(theta)*np.cos(phi), r*np.sin(theta)*np.sin(phi), r*np.cos(theta)])
        pos = rel + center_pos # cartesian satellite positions in unwrapped frame
        return pos, v, m # v is either 3D peculiar velocity or line-of-sight velocity

    def place_centrals(self, n_groups=3, radius=10, sim=False):
        """
        Place n_groups centrals equidistantly around the origin.
        They are placed on a circle at equal angular intervals.
        
        Args:
            n_groups: Number of groups to place
            radius: Distance from origin (in Mpc)
            
        Returns:
            masses, positions, vlos, radii, vvirs, n_groups
        """
        masses = np.random.uniform(9.5, 11.5, n_groups)
        # Sort so most massive first
        order = np.argsort(-masses)
        masses = masses[order]
        
        pos = np.zeros((n_groups, 3))
        if sim:
            v_cen = np.zeros((n_groups, 3))
        else:
            v_cen = np.zeros(n_groups)
        radii = np.zeros(n_groups)
        vvirs = np.zeros(n_groups)
        r_cen = np.random.uniform(0.2*radius, 0.8*radius, n_groups)
        z_cen = z_at_value(cosmo.comoving_distance, r_cen * u.Mpc).value
        
        # Place groups equidistantly on a circle
        for i in range(n_groups):
            phi = 2 * np.pi * i / n_groups
            theta = np.random.uniform(0., 2*np.pi)
            pos[i, 0] = r_cen[i] * np.cos(phi) * np.sin(theta)
            pos[i, 1] = r_cen[i] * np.sin(phi) * np.sin(theta)
            pos[i, 2] = r_cen[i] * np.cos(theta) # z is free parameter
            
            # Generate halo properties
            _, R_h, v_vir = self.halo_props(z_cen[i], masses[i])
            radii[i] = R_h
            vvirs[i] = v_vir
            if sim:
                v_cen[i] = np.random.uniform(-50., 50., 3).tolist()
            # Line-of-sight velocity (radial from origin)
            else:
                v_cen[i] = np.random.uniform(-50., 50.)
        
        return masses, pos, v_cen, radii, vvirs, n_groups # cartesian positions in unwrapped frame
    

    def assign_satellite_counts(self, n_groups=3, n_sats=20):
        N_sat = np.zeros(n_groups, dtype=int)
        # randomly distribution n_sats among n_groups
        for _ in range(n_sats):
            gid = np.random.randint(0, n_groups)
            N_sat[gid] += 1
        return N_sat
        
    def build_obs_catalog(self, n_groups=3, n_sats=20, redshift=False):
        """
        Build a data structure matching the observational GroupFinder input schema:
        positions: list[[dist, dec, ra]] in Mpc, degrees or (redshift, degrees)
        velocities: list[v_los] in km/s (written to [0.] if in redshift mode)
        masses: list[log10 M*]
        ids: list[int]
        group_indices: list[list[int]]
        """

        masses_cen, pos_cen, v_cen, _, _, n_groups = self.place_centrals(n_groups=n_groups, radius=self.box_size)
        N_sat = self.assign_satellite_counts(n_groups=n_groups, n_sats=n_sats)

        all_ids = []
        all_masses = []
        all_cart = []
        all_sph = []
        all_vel = []
        group_indices = []
        gid = 0

        for i in range(n_groups):
            group = []
            # central
            cid = gid
            group.append(cid)
            all_ids.append(cid)
            all_masses.append(float(masses_cen[i]))
            all_cart.append(pos_cen[i].tolist())
            r_c, th_c, ph_c = self.cart_to_sph(pos_cen[i].tolist())
            dec_c = np.pi/2 - th_c
            all_sph.append([float(r_c), float(dec_c), float(ph_c)])
            all_vel.append(float(v_cen[i]))
            gid += 1

            sat_pos, sat_v, sat_m = self.generate_satellites(masses_cen[i], pos_cen[i], v_cen[i], N_sat[i])
            for j in range(sat_pos.shape[0]):
                sid = gid
                group.append(sid)
                all_ids.append(sid)
                all_masses.append(float(sat_m[j]))
                all_cart.append(sat_pos[j].tolist())
                r_s, th_s, ph_s = self.cart_to_sph(sat_pos[j].tolist())
                dec_s = np.pi/2 - th_s
                all_sph.append([float(r_s), float(dec_s), float(ph_s)])
                all_vel.append(float(sat_v[j]))
                gid += 1

            group_indices.append(group)

        # Integrity: central mass > satellite mass
        for group in group_indices:
            if len(group) > 1:
                cm = all_masses[group[0]]
                assert all(all_masses[s] < cm - 1e-6 for s in group[1:])

        if redshift:
            all_sph = np.array(all_sph)
            dists = all_sph[:,0]
            zs = z_at_value(cosmo.comoving_distance, dists*u.Mpc).value
            all_sph[:,0] = zs
            all_vel = [0.]
            all_sph = all_sph.tolist()
        data = {
            "masses": all_masses,
            "positions": all_sph,
            "ids": all_ids,
            "velocities": all_vel,
            "group_indices": group_indices
        }
        return data

    def build_sim_catalog(self, groups_per_mw=3, n_sats=20, origin=False):
        """
        Build a data structure matching the simulated GroupFinder input schema:
        mass_vec: list[log10 M*]
        positions_vec: list[[x,y,z]]
        gc_idx_vec: list[int]
        MW_ID_pos_vec: list[x,y,z] 
        MW_vels_vec: list[vx,vy,vz] 
        stellar_mass_all_vec: flat array sized (1+max global id)
        velocities_vec: list[[vx,vy,vz]]
        """
        mw_masses = []
        mw_positions = []
        mw_velocities = []
        mw_ids = []
        group_indices = []
        global_id = 0
        stellar_mass_all_vec = []

        masses_cen, pos_cen, v_cen, radii, vvir, groups_per_mw = self.place_centrals(n_groups=groups_per_mw, radius=self.box_size/2, sim=True)
        N_sat = self.assign_satellite_counts(n_groups=groups_per_mw, n_sats=n_sats)
        
        if origin:
            MW_ID_pos_vec = [0., 0., 0.]
            MW_ID_vel_vec = [0., 0., 0.]
        else:
            MW_ID_pos_vec = np.random.uniform(-self.box_size/2, self.box_size/2, 3).tolist()
            MW_ID_vel_vec = np.random.uniform(-50., 50., 3).tolist()

        for i in range(groups_per_mw):
            group = []
            # Central
            cid = global_id
            group.append(cid)
            global_id += 1
            mw_masses.append(float(masses_cen[i]))
            # Groups were placed in unwrapped frame; we need to wrap them
            # First shift wrt MW position
            pos_absolute = pos_cen[i] + np.array(MW_ID_pos_vec)
            mw_velocities.append([float(v_cen[i][0]), float(v_cen[i][1]), float(v_cen[i][2])])
            mw_ids.append(cid)
            if cid >= len(stellar_mass_all_vec):
                # extend
                stellar_mass_all_vec.extend([0.0] * (cid - len(stellar_mass_all_vec) + 1))
            stellar_mass_all_vec[cid] = masses_cen[i]
                
            # Apply PBC wrapping to get final central position
            pos_wrapped = np.array([(pos_absolute[j] % self.box_size) for j in range(3)])
            mw_positions.append(pos_wrapped.tolist())

            # Satellites: generate around the unwrapped central position 
            sat_pos, sat_v, sat_m = self.generate_satellites(masses_cen[i], pos_cen[i], v_cen[i], N_sat[i], sim=True)
            for j in range(sat_pos.shape[0]):
                sid = global_id
                group.append(sid)
                global_id += 1
                mw_masses.append(float(sat_m[j]))
                # Satellite position is around unwrapped central, just need to wrap it and shift out of MW frame
                sat_pos_wrapped = np.array([(sat_pos[j][k]+np.array(MW_ID_pos_vec)[k]) % self.box_size for k in range(3)])
                mw_velocities.append([float(sat_v[j][0]), float(sat_v[j][1]), float(sat_v[j][2])])
                mw_ids.append(sid)
                if sid >= len(stellar_mass_all_vec):
                    stellar_mass_all_vec.extend([0.0] * (sid - len(stellar_mass_all_vec) + 1))
                stellar_mass_all_vec[sid] = sat_m[j]
                # Store wrapped satellite position
                mw_positions.append(sat_pos_wrapped.tolist())
            group_indices.append(group)

        data = {
            "masses": mw_masses,
            "positions": mw_positions,
            "ids": mw_ids,
            "ref_positions": np.array(MW_ID_pos_vec).reshape(1, 3),
            "ref_velocities": np.array(MW_ID_vel_vec).reshape(1, 3),
            "velocities": mw_velocities,
            "group_indices": group_indices
        }
        return data

    @staticmethod
    def write_h5(data, outfile: str):
        # Make parent directory if it doesn't exist
        Path(outfile).parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(outfile, "w") as f:
            for key, value in data.items():
                if key != "group_indices":
                    f.create_dataset(name=key, data=value)
                else: # write variable length array
                    dt = h5py.special_dtype(vlen=np.dtype('int'))
                    # Convert list of lists to array of numpy arrays
                    value_array = np.empty(len(value), dtype=object)
                    for i, group in enumerate(value):
                        value_array[i] = np.array(group, dtype=int)
                    f.create_dataset(name=key, data=value_array, dtype=dt)
        print(f"Input data written to: {outfile}")

    def create_test_data(self, type: str, outfile: str, n_groups=3, n_sats=20, origin=False, redshift=False):
        if type == "obs":
            data = self.build_obs_catalog(n_groups=n_groups, n_sats=n_sats, redshift=redshift)
        elif type == "sim":
            data = self.build_sim_catalog(groups_per_mw=n_groups, n_sats=n_sats, origin=origin)
        else:
            raise ValueError("Type must be 'obs' or 'sim'")

        self.write_h5(data, outfile)

def check_results(input_file: str, result_file: str):
    with h5py.File(input_file, "r") as f:
        input_group_indices = f['group_indices'][:]
        #print(input_group_indices)

    with h5py.File(result_file, "r") as f:
        member_ids = f['results']['group_member_ids'][:]
        offsets = f['results']['group_member_offsets'][:]
    found_group_indices = [member_ids[offsets[i]:offsets[i+1]] for i in range(len(offsets)-1)]
    #print(found_group_indices)

    count_nonmatches = 0
    for input_group, found_group in zip(input_group_indices, found_group_indices):
        input_set = set(sorted(input_group))
        found_set = set(sorted(found_group))
        if input_set != found_set:
            # The last element of each input group (highest ID) is the isolated satellite
            # which should NOT be found by the group finder
            if len(input_set) > 1:
                input_isolated = max(input_set)
                input_set.remove(input_isolated)
                # Check if the remaining members match
                if input_set != found_set:
                    print(f"Group members do not match for group with central ID {input_group[0]}")
                    print(f"  Expected (excluding isolated): {np.array(list(input_set), dtype=int)}")
                    print(f"  Found: {np.array(list(sorted(found_set)), dtype=int)}")
                    count_nonmatches += 1
            else:
                print(f"Group members do not match for group with central ID {input_group[0]}")
                count_nonmatches += 1
    print("Results checked.")
    if count_nonmatches == 0:
        print("All groups match expected members (excluding isolated satellites).")
        return True
    else:
        print(f"{count_nonmatches} groups did not match expected members.")
        return False

def plot_groups(input_file: str, output_file: str, type: str):
    # Visualize the groups from input and output files
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    with h5py.File(input_file, "r") as f:
        positions = f['positions'][:]
        group_indices_i = f['group_indices'][:]

    with h5py.File(output_file, "r") as f:
        member_ids = f['results']['group_member_ids'][:]
        offsets = f['results']['group_member_offsets'][:]
    group_indices_o = [member_ids[offsets[i]:offsets[i+1]] for i in range(len(offsets)-1)]

    if type == "obs":
        # Convert spherical to Cartesian for plotting
        positions = np.array([[
            pos[0] * np.cos(np.radians(pos[2])) * np.cos(np.radians(pos[1])),
            pos[0] * np.cos(np.radians(pos[2])) * np.sin(np.radians(pos[1])),
            pos[0] * np.sin(np.radians(pos[2]))
        ] for pos in positions])
    elif type == "sim":
        pass
    else:
        raise ValueError("Type must be 'obs' or 'sim'")

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    colors_i = plt.cm.get_cmap('tab10', len(group_indices_i))
    colors_o = plt.cm.get_cmap('Set1', len(group_indices_o))

    for i, group in enumerate(group_indices_i):
        group_positions = positions[group]
        ax.scatter(group_positions[:,0], group_positions[:,1], group_positions[:,2], color=colors_i(i), label=f'Group {i}: Input', marker='o')

    for i, group in enumerate(group_indices_o):
        group_positions = positions[group]
        ax.scatter(group_positions[:,0], group_positions[:,1], group_positions[:,2], color=colors_o(i), label=f'Group {i}: Found', marker='x')
    ax.set_xlabel('X [Mpc]')
    ax.set_ylabel('Y [Mpc]')
    ax.set_zlabel('Z [Mpc]')
    ax.set_title('Visualization of Groups: Input vs Found')
    plt.show()