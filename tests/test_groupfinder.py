"""
Run the observational and simulation density contrast configurations and the 6D
simulation configurations N times to verify correctness of group finder implementation.
For density contrast runs, pass if more than 98% tests pass (because of probabilistic
nature of the group assignment, which can split groups into subgroups occasionally).
(for 6D sim runs, require 100% pass rate).
"""
from gen_data import GroupFinderTest, check_results
import sys
import os
import h5py
import numpy as np
import pytest
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
from groupfinder_interface import GroupFinderInterface, SimulationData, ObservationalData, GroupFinderRunner, run_groupfinder
from input import InterpolationData

PASS_PERCENTAGE = 0.98
NUM_TESTS = 100

def suppress_output():
    """Suppress stdout and keep terminal clean"""
    sys.stdout.flush()
    old_stdout = os.dup(1)
    devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, 1)
    os.close(devnull)
    return old_stdout

def restore_output(old_stdout):
    """Restore output to terminal"""
    sys.stdout.flush()
    os.dup2(old_stdout, 1)
    os.close(old_stdout)

class Tests:
    def test_interpolation_data_generation(self, max_z = 0.1):
        # Run halo concentration and z-distance data generation once, using same cosmology for all tests
        InterpolationData.generate_concentration_data(max_z=max_z)
        InterpolationData.generate_z_dist_data(max_z=max_z)
    
    def test_sim_yang05(self, n_tests=NUM_TESTS):
        passed_sim = 0
        print("Running sim_config tests...")
        old_stdout = suppress_output()
        interface_sim = GroupFinderInterface()
        runner = GroupFinderRunner()

        # Set up config once for all sim tests
        interface_sim.R_h_group_val = 1.0
        interface_sim.V_vir_group_val = 3.0
        interface_sim.R_h_iso_val = 2.0
        interface_sim.V_vir_iso_val = 3.0
        interface_sim.contrast_val = True
        interface_sim.dim = 3
        interface_sim.tree_search_val = False
        interface_sim.sat_reclass_val = True
        interface_sim.iso_reclass_val = True
        interface_sim.B_scaling = 1.
        interface_sim.h = 0.6774
        interface_sim.omega_M = 0.3089
        interface_sim.periodic = True
        interface_sim.box_size = 35000./interface_sim.h/1000.  # in Mpc
        interface_sim.R_max = interface_sim.box_size * np.sqrt(3)/2.  # half the box diagonal: max possible distance in periodic box
        interface_sim.config(os.path.join(parent_dir, "input/sim_config.json"), obs=False)

        for i in range(n_tests):
            # Generate simulation test data
            test_sim = GroupFinderTest(box_size=interface_sim.box_size, h=interface_sim.h, omega_M=interface_sim.omega_M)
            test_sim.create_test_data(type="sim", outfile=os.path.join(parent_dir, "input/data/sim_data.h5"), n_groups=3)
            
            run_groupfinder('sim', os.path.join(parent_dir, 'input/data/sim_data.h5'), os.path.join(parent_dir, 'sim_gf_result.h5'), os.path.join(parent_dir, 'input/sim_config.json'))
            res = check_results(os.path.join(parent_dir, 'input/data/sim_data.h5'), os.path.join(parent_dir, 'sim_gf_result.h5'))
            if res:
                passed_sim += 1

            os.remove(os.path.join(parent_dir, "input/data/sim_data.h5"))
            os.remove(os.path.join(parent_dir, "sim_gf_result.h5"))
            
        restore_output(old_stdout)
        print(f"{passed_sim} out of {n_tests} tests passed.")
        passed_percentage = passed_sim/n_tests
        assert passed_percentage >= PASS_PERCENTAGE, "Overall sim_config test FAILED."

    def test_obs_yang05(self, n_tests=NUM_TESTS):
        passed_obs = 0
        print("Running obs_config tests...")
        old_stdout = suppress_output()
        for i in range(n_tests):
            # Reconfigured each time due to B scaling calculation
            interface = GroupFinderInterface()
            runner = GroupFinderRunner()

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

            # Generate observational test data
            test = GroupFinderTest(box_size=interface.R_max, h=interface.h, omega_M=interface.omega_M)
            test.create_test_data(type="obs", outfile=os.path.join(parent_dir, "input/data/obs_data.h5"))

            with h5py.File(os.path.join(parent_dir, "input/data/obs_data.h5"), "r") as f:
                obs_positions = np.array(f['positions'][:]) # already in spherical coords (dist [Mpc], ra [rad], dec [rad])
                obs_velocities = np.array(f['velocities'][:]) # heliocentric peculiar velocities [km/s]
                obs_masses = np.array(f['masses'][:]) # stellar masses [log10(M_sun)]
                obs_ids = np.array(f['ids'][:])
                obs_group_indices = np.array(f['group_indices'][:])
            obs_data = ObservationalData(positions=obs_positions,
                                        velocities=obs_velocities,
                                        masses=obs_masses,
                                        ids=obs_ids,
                                        group_indices=obs_group_indices)
            
            # Generate simulation test data for B parameter calculation
            test_sim = GroupFinderTest(box_size=35/interface.h, h=interface.h, omega_M=interface.omega_M)
            test_sim.create_test_data(type="sim", outfile=os.path.join(parent_dir, "input/data/sim_data.h5"), n_groups=4)

            with h5py.File(os.path.join(parent_dir, "input/data/sim_data.h5"), "r") as f:
                sim_positions = np.array(f['positions'][:]) # in physical Cartesian box coords [Mpc]
                sim_velocities = np.array(f['velocities'][:]) # in physical Cartesian box coords [km/s]
                sim_masses = np.array(f['masses'][:]) # in physical log stellar masses [log10(M_sun)]
                sim_ids = np.array(f['ids'][:])
                sim_ref_positions = np.array(f['ref_positions'][:])
                sim_ref_velocities = np.array(f['ref_velocities'][:])

            sim_data = SimulationData(positions=sim_positions,
                                    velocities=sim_velocities,
                                    masses=sim_masses,
                                    ids=sim_ids,
                                    ref_positions=sim_ref_positions,
                                    ref_velocities=sim_ref_velocities)
                
            interface.calculate_B(sim_data=sim_data,
                                obs_data=obs_data,
                                mass_limit=6.0,
                                R_sim=35.0/interface.h,
                                R_obs=50.0)

            interface.config(os.path.join(parent_dir, "input/obs_config.json"), obs=True)

            run_groupfinder('obs', os.path.join(parent_dir, "input/data/obs_data.h5"), os.path.join(parent_dir, "obs_gf_result.h5"), os.path.join(parent_dir, "input/obs_config.json"))
            res = check_results(os.path.join(parent_dir, "input/data/obs_data.h5"), os.path.join(parent_dir, "obs_gf_result.h5"))
            if res:
                passed_obs += 1

            os.remove(os.path.join(parent_dir, "input/data/obs_data.h5"))
            os.remove(os.path.join(parent_dir, "input/data/sim_data.h5"))
            os.remove(os.path.join(parent_dir, "obs_gf_result.h5"))

        restore_output(old_stdout)
        print(f"{passed_obs} out of {n_tests} tests passed.")
        passed_percentage = passed_obs/n_tests
        assert passed_percentage >= PASS_PERCENTAGE, "Overall obs_config test FAILED."

    def test_sim_6D(self, n_tests=NUM_TESTS):
        passed_sim6D = 0
        print("Running sim_6D tests...")
        old_stdout = suppress_output()

        interface_sim6D = GroupFinderInterface()
        runner = GroupFinderRunner()

        # Set up config once for all sim tests
        interface_sim6D.R_h_group_val = 1.0
        interface_sim6D.V_vir_group_val = 3.0
        interface_sim6D.R_h_iso_val = 2.0
        interface_sim6D.V_vir_iso_val = 3.0
        interface_sim6D.contrast_val = False
        interface_sim6D.dim = 6
        interface_sim6D.tree_search_val = True
        interface_sim6D.sat_reclass_val = True
        interface_sim6D.iso_reclass_val = True
        interface_sim6D.B_scaling = 1.
        interface_sim6D.h = 0.6774
        interface_sim6D.omega_M = 0.3089
        interface_sim6D.periodic = True
        interface_sim6D.box_size = 35000./interface_sim6D.h/1000.  # in Mpc
        interface_sim6D.R_max = interface_sim6D.box_size * np.sqrt(3)/2.  # half the box diagonal: max possible distance in periodic box
        interface_sim6D.config(os.path.join(parent_dir, "input/sim_config_6D.json"), obs=False)

        for i in range(n_tests):
            # Generate simulation test data
            test_sim6D = GroupFinderTest(box_size=interface_sim6D.box_size, h=interface_sim6D.h, omega_M=interface_sim6D.omega_M)
            test_sim6D.create_test_data(type="sim", outfile=os.path.join(parent_dir, "input/data/sim_data.h5"), n_groups=3)
            
            run_groupfinder('sim', os.path.join(parent_dir, 'input/data/sim_data.h5'), os.path.join(parent_dir, 'sim_gf_result.h5'), os.path.join(parent_dir, 'input/sim_config_6D.json'))
            res = check_results(os.path.join(parent_dir, 'input/data/sim_data.h5'), os.path.join(parent_dir, 'sim_gf_result.h5'))
            if res:
                passed_sim6D += 1

            os.remove(os.path.join(parent_dir, "input/data/sim_data.h5"))
            os.remove(os.path.join(parent_dir, "sim_gf_result.h5"))

        restore_output(old_stdout)
        print(f"{passed_sim6D} out of {n_tests} tests passed.")
        passed_percentage = passed_sim6D/n_tests
        assert passed_percentage == 1.0, "Overall sim_config_6D test FAILED." # this is by design exact
        
    def test_obs_redshift_yang05(self, n_tests=NUM_TESTS):
        passed_obs = 0
        print("Running tests of 'redshift survey' observational mode...")
        old_stdout = suppress_output()
        for i in range(n_tests):
            # Reconfigured each time due to B scaling calculation
            interface = GroupFinderInterface()
            runner = GroupFinderRunner()

            # Set up config
            interface.R_h_group_val = 1.0
            interface.V_vir_group_val = 3.0
            interface.R_h_iso_val = 2.0
            interface.V_vir_iso_val = 3.0
            interface.contrast_val = True
            interface.sat_reclass_val = True
            interface.iso_reclass_val = True
            interface.R_max = 350.0
            interface.h = 0.6774
            interface.omega_M = 0.3089
            interface.use_distance = False
            interface.tree_search = True

            # Generate observational test data
            test = GroupFinderTest(box_size=interface.R_max, h=interface.h, omega_M=interface.omega_M)
            test.create_test_data(type="obs", outfile=os.path.join(parent_dir, "input/data/obs_data.h5"), n_groups=6, redshift=True)

            with h5py.File(os.path.join(parent_dir, "input/data/obs_data.h5"), "r") as f:
                obs_positions = np.array(f['positions'][:]) # already in (redshift, ra [rad], dec [rad])
                obs_masses = np.array(f['masses'][:]) # log stellar masses [log10(M_sun)]
                obs_ids = np.array(f['ids'][:])
                obs_group_indices = np.array(f['group_indices'][:])
            obs_data = ObservationalData(positions=obs_positions,
                                        masses=obs_masses,
                                        ids=obs_ids,
                                        group_indices=obs_group_indices)
            
            # Generate simulation test data for B parameter calculation (do not need to make new config)
            test_sim = GroupFinderTest(box_size=300, h=interface.h, omega_M=interface.omega_M)
            test_sim.create_test_data(type="sim", outfile=os.path.join(parent_dir, "input/data/sim_data.h5"), n_groups=7, origin=True, redshift=True)

            with h5py.File(os.path.join(parent_dir, "input/data/sim_data.h5"), "r") as f:
                sim_positions = np.array(f['positions'][:]) # in physical Cartesian box coords [Mpc]
                sim_velocities = np.array(f['velocities'][:]) # in physical Cartesian box coords [km/s]
                sim_masses = np.array(f['masses'][:]) # in physical log stellar masses [log10(M_sun)]
                sim_ids = np.array(f['ids'][:])

            sim_data = SimulationData(positions=sim_positions,
                                    velocities=sim_velocities,
                                    masses=sim_masses,
                                    ids=sim_ids)
                
            interface.calculate_B(sim_data=sim_data,
                                obs_data=obs_data,
                                mass_limit=6.0,
                                R_sim=300,
                                R_obs=interface.R_max)

            interface.config(os.path.join(parent_dir, "input/obs_config_z.json"), obs=True)

            run_groupfinder('obs', os.path.join(parent_dir, "input/data/obs_data.h5"), os.path.join(parent_dir, "obs_gf_result.h5"), os.path.join(parent_dir, "input/obs_config_z.json"))
            res = check_results(os.path.join(parent_dir, "input/data/obs_data.h5"), os.path.join(parent_dir, "obs_gf_result.h5"))
            if res:
                passed_obs += 1

            os.remove(os.path.join(parent_dir, "input/data/obs_data.h5"))
            os.remove(os.path.join(parent_dir, "input/data/sim_data.h5"))
            os.remove(os.path.join(parent_dir, "obs_gf_result.h5"))

        restore_output(old_stdout)
        print(f"{passed_obs} out of {n_tests} tests passed.")
        passed_percentage = passed_obs/n_tests
        assert passed_percentage >= PASS_PERCENTAGE, "Overall obs_config redshift test FAILED."