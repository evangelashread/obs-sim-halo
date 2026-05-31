import os
from colossus.cosmology import cosmology
import astropy.cosmology
from colossus.halo import concentration
import numpy as np
import h5py
import shutil
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Adjust as needed
cosmology.setCosmology('planck15')
colossus_cosmo = cosmology.getCurrent()
astropy_cosmo = astropy.cosmology.Planck15

def generate_concentration_data(min_z=0.0, max_z=0.35, zstep=0.01, 
                                min_Mh=8.0, max_Mh=15.0, Mhstep=0.1):
    """Generate concentration-mass relation data and save to HDF5 file.
    Needed for group finder interpolation that goes into density contrast.

    Parameters:
        min_z: float
            Minimum redshift.
        max_z: float
            Maximum redshift.
        zstep: float
            Spacing between redshift bins.
        min_Mh: float
            Minimum halo mass in log10(Msun).
        max_Mh: float
            Maximum halo mass in log10(Msun).
        Mhstep: float
            Spacing between halo mass bins.

    Returns
        None
    """
    outfile_concentration = 'concentration.h5'
    
    halo_masses = np.arange(min_Mh, max_Mh + Mhstep, Mhstep)
    z_vals = np.arange(min_z, max_z + zstep, zstep)
    n_z = len(z_vals)
    width = len(str(n_z - 1))
    h = colossus_cosmo.H0 / 100.0

    # compute concentration as a function of mass and redshift
    c200c = [] 
    for z in z_vals:
        c200c_z = concentration.concentration((10**halo_masses)/h, '200c', z, model = 'diemer19')
        c200c.append(c200c_z) # length N * number of redshift bins

    with h5py.File(outfile_concentration, 'w') as f:
        f.create_dataset('halo_masses', data=halo_masses)
        f.create_dataset('redshifts', data=z_vals)
        grp = f.create_group('concentration')
        for i, arr in enumerate(c200c):
            name = f"D{i:0{width}d}"
            dset = grp.create_dataset(str(name), data=arr)
            dset.attrs['z'] = z_vals[i]
    print(f'Wrote concentration-mass relation to {outfile_concentration}')
    shutil.move(outfile_concentration, os.path.join(parent_dir, f'input/{outfile_concentration}'))


def generate_z_dist_data(min_z=0.0, max_z=0.35, zstep=1e-4):
    """Generate redshift-distance relation data and save to HDF5 file.
    This allows for fast interpolation in the group finder when converting
    between comoving distances and redshifts.

    Parameters:
        min_z: float
            Minimum redshift.
        max_z: float
            Maximum redshift.
        zstep: float
            Spacing between redshift bins.

    Returns
        None
    """
    outfile = 'redshift_distance.h5'
    
    z_vals = np.arange(min_z, max_z + zstep, zstep)
    cMpc = astropy_cosmo.comoving_distance(z_vals).value # in Mpc

    with h5py.File(outfile, 'w') as f:
        f.create_dataset('distances', data=cMpc)
        f.create_dataset('redshifts', data=z_vals)
    print(f'Wrote redshift-distance relation to {outfile}')
    shutil.move(outfile, os.path.join(parent_dir, f'input/{outfile}'))