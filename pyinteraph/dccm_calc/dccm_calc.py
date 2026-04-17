#!/usr/bin/env python

import MDAnalysis as mda
import numpy as np
from MDAnalysis.analysis import align
from matplotlib import pyplot as plt
import argparse
import sys

def fluct_calc(u):
    """ 
    Collects coordinates of all alpha carbons of each frame and calculates its fluctuation
    (Each row is a frame and each column is the fluctuation of the residue)
    """
    ca = u.select_atoms("name CA")
    coords = []
    for ts in u.trajectory:
        coords.append(ca.positions.copy())

    coords = np.array(coords)
    mean_pos = coords.mean(axis=0)
    fluct = coords - mean_pos
    return fluct

def dccm_calc(fluct):
    """ Generates n*n matrix showing the paired movement of the n residues """
    n_frames, n_res, _ = fluct.shape

    # making the columns into frames and then flattening so that each row is a time series of the motion of each residue
    X = fluct.transpose(1, 0, 2).reshape(n_res, n_frames * 3)
    cov = (X @ X.T) / (n_frames*3) # covariance b/w pairs of residues using matrix multiplication
    var = np.diag(cov)
    dccm = cov / np.sqrt(np.outer(var, var))
    return dccm

def main():
    parser = argparse.ArgumentParser(description="Compute DCCM from MD trajectory")
    parser.add_argument("pdb", help="Topology file (e.g. .pdb)")
    parser.add_argument("trajectory", help="Trajectory file (e.g .xtc)")

    args = parser.parse_args()

    # Loading the structure and trajectory
    try:
        u = mda.Universe(args.pdb, args.trajectory)
    except FileNotFoundError:
        print("Error: One or more files not found")
        sys.exit(1)
    except ValueError:
        print("Error: Invalid file format for pdb or trajectory")
        sys.exit(1)

    # Aligning the frames w.r.t C-alpha
    align.AlignTraj(u, u, select="name CA", in_memory=True).run()

    fluct = fluct_calc(u)
    dccm = dccm_calc(fluct)

    np.savetxt("dccm.csv", dccm, delimiter=",")  # writing to a csv

    # Visualization
    plt.imshow(dccm, cmap='bwr', vmin=-1, vmax=1, origin='lower') 
    plt.colorbar(label='Correlation') 
    plt.title("DCCM") 
    plt.xlabel("Residue number") 
    plt.ylabel("Residue number") 
    plt.savefig('dccm.png')

if __name__ == "__main__":
    main()
