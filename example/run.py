import time
import numpy
from ase.io import iread
from mace.calculators import MACECalculator

dataset = 'dataset.xyz'
model_path = '../models/SPICE_medium_dipole.model'
device = 'cpu'
default_dtype = 'float64'
model_type = 'DipoleMACE'
calc = MACECalculator(model_path, device=device, default_dtype=default_dtype, model_type=model_type)

print ('%20s %20s %15s' % ('reference dipole [D]', 'predicted dipole [D]', 'time [ms]') )
for atoms in iread(dataset):
    start_time = time.time()
    calc.get_property('dipole', atoms=atoms)
    print('%20.5e %20.5e % 15.5f' % (numpy.linalg.norm(calc.results['dipole']), numpy.linalg.norm(atoms.info['REF_dipole']), (-start_time + time.time()) * 1000 ))
