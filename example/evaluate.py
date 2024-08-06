#!/usr/bin/env python
import time
import numpy
import numpy as np
import csv
import argparse
from ase.io import iread
from mace.calculators import MACECalculator
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.stats import norm
import seaborn as sns

au_to_ang = 0.529177
au_to_debye = 2.54174692380177980056

###################################################################################################
# Command Line Arguments
###################################################################################################
parser = argparse.ArgumentParser(description='Evaluate molecular dipoles.')
parser.add_argument('-c', '--csv', type=str, default='output.csv', help='Output CSV file name')
parser.add_argument('--plot', action='store_true', help='Plot the results as violin plot(s)')
parser.add_argument('--split', type=int, nargs='*', help='Split indices for dipole data in plotting settings')
parser.add_argument('database', type=str, nargs='?', default='dataset.xyz', help='Input .xyz database file')
parser.add_argument('-m','--maxdev', type=float, default=100.0, help='Define maximum deviation of regularized dipole difference to determine outliers ')
args = parser.parse_args()

###################################################################################################
# User-defined setup
###################################################################################################

dataset = args.database
model_path = '../models/SPICE_small_dipole.model'
# if you want to use the medium model
# model_path = '../models/SPICE_medium_dipole.model'
device = 'cpu'
default_dtype = 'float64'
model_type = 'DipoleMACE'
calc = MACECalculator(model_path, device=device, default_dtype=default_dtype, model_type=model_type)

#outlier cut-off
outlier=args.maxdev


###################################################################################################
# Helper routines
###################################################################################################

def mhg_delta(tref, tdip):
    # Compute the regularized dipole error measure according to Hait and Head-Gordon
    return (tdip - tref) / max(tref, 1.0)

###################################################################################################
# Evaluation loop
###################################################################################################

timings = 0
dipole = []
dipole_delta = []
names=[]
outliers=[]

print(f'\nCalculated dipole components will be written to {args.csv} (in a.u.)\n')

print('%30s %22s %22s %15s' % ('System', 'reference dipole [D]', 'predicted dipole [D]', 'time [ms]'))

with open(args.csv, 'w', newline='') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(['Name', 'Dip_x', 'Dip_y', 'Dip_z'])
    for i,atoms in enumerate(iread(dataset)):
        start_time = time.time()
        calc.get_property('dipole', atoms=atoms)
        mstime = (time.time() - start_time) * 1000
        timings += mstime / 1000
        name = atoms.info.get('name', f'molecule_{i}')
        dip = calc.results['dipole'] / au_to_ang
        ref = atoms.info['REF_dipole']
        tot_dip = numpy.linalg.norm(dip) * au_to_debye
        tot_ref = numpy.linalg.norm(ref) * au_to_debye
        #dipole.append(np.array(dip))
        #names.append(name)
        delta=mhg_delta(tot_ref, tot_dip)
        if delta < outlier:
           dipole_delta.append(delta)
        else:
           outliers.append(name)
        csvwriter.writerow([name, *dip])
        print('%30s %22.5f %22.5f % 15.5f' % (name, tot_ref, tot_dip, mstime))

print(f'\nTotal wall-time : {timings:.3f} s')
print('\n--------------------------------------------------------------------------------------------')
print('Regularized dipole moment error measures (see Hait & Head-Gordon, JCTC, 2018, 14, 1969–1981)')

mean = np.mean(dipole_delta)
mad = np.mean(np.abs(dipole_delta - mean))
rmse = np.sqrt(np.mean(np.square(dipole_delta - mean)))
print()
print(f'mean: {mean:.4f}')
print(f'MAD : {mad:.4f}')
print(f'RMSE: {rmse:.4f}')
print()
if len(outliers) > 0:
   print(f'{len(outliers)} outliers with regularized dipole moment > {outlier} :')
   print(outliers)

print('--------------------------------------------------------------------------------------------')

###################################################################################################
# Plotting section
###################################################################################################

if args.plot:
    print('Preparing plot...')

    # Define the splitting indices from the command line argument or use default
    split_indices = args.split if args.split else [len(dipole_delta)]
    #print(f'{split_indices}')

    ddip = []
    prev_index = 0
    for index in split_indices:
        upper=prev_index+index
        ddip.append(np.array(dipole_delta[prev_index:upper]))
        #print(f'{names[prev_index:upper]}')
        prev_index=upper
    if prev_index < len(dipole_delta):
        ddip.append(np.array(dipole_delta[prev_index:]))
    # Print the lengths of each array in ddip
    for i, subset in enumerate(ddip):
        print(f'Size of subset-{i+1}: {len(subset)}')

    # Labels for the x-ticks and colors
    labels = [f'subset-{i+1}' for i in range(len(ddip))]
    colors = sns.color_palette("husl", len(ddip))

    # Adjust figure size based on number of subsets
    fig_width = 4 + 0.5 * len(ddip)
    fig, ax = plt.subplots(figsize=(fig_width, 5))

    # Create the violin plot
    sns.violinplot(data=ddip, ax=ax, linewidth=1, palette=colors)

    # Setting the x-tick labels to the names of the arrays
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels)

    # Adding title, labels, and other things (optional)
    ax.set_ylabel('Regularized dipole moments', fontsize=12)

    ax.text(0.02, 0.97, f'n = {len(dipole_delta)}', transform=ax.transAxes,
            verticalalignment='top', horizontalalignment='left',
            fontsize=10, color='black', bbox=dict(facecolor='white', alpha=0.5))

    ax.yaxis.grid(True, linestyle='--', linewidth=0.5, color='lightgray')

    plt.tight_layout()

    # Save and display the plot
    plt.savefig('violin.svg')
    plt.savefig('violin.pdf', dpi=600)
    plt.show()

