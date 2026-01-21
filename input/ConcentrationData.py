import os
from colossus.cosmology import cosmology
cosmology.setCosmology('planck15') # Adapt as needed
from colossus.halo import concentration
import numpy as np
import h5py
import shutil
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def generate_concentration_data(h=0.6774, halo_masses=np.linspace(9.0, 17.0, 801)):
    """Generate concentration-mass relation data and save to HDF5 file.
    Needed for group finder interpolation that goes into density contrast.

    Parameters:
        h: float
            Dimensionless Hubble parameter.
        halo_masses: array_like
            Array of halo masses in log10(Msun/h).

    Returns
        None
    """
    outfile_concentration = 'concentration.h5'

    # compute concentration as a function of mass
    c200c = concentration.concentration(10**halo_masses/h, '200c', 0, model = 'diemer19')  # length N

    with h5py.File(outfile_concentration, 'w') as f:
        f.create_dataset('halo_masses', data=halo_masses)
        f.create_dataset('concentration', data=c200c)
    print(f'Wrote concentration-mass relation to {outfile_concentration}')
    shutil.move('concentration.h5', os.path.join(parent_dir, 'input/concentration.h5'))