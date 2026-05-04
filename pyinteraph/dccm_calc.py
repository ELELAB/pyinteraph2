#!/usr/bin/env python

import MDAnalysis as mda
import numpy as np
from MDAnalysis.analysis import align
from matplotlib import pyplot as plt
import argparse
import sys

def fluct_calc(coords):
    """
    Collects coordinates of all alpha carbons of each frame and calculates its fluctuation
    (Each row is a frame and each column is the fluctuation of the residue)
    """
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
    parser = argparse.ArgumentParser(description="Compute correlation metrics from MD trajectory")
    parser.add_argument("pdb", help="Topology file (e.g. .pdb)")
    parser.add_argument("trajectory", help="Trajectory file (e.g .xtc)")
    parser.add_argument(
        "method",
        choices = ["dccm", "lmi"],
        help = "Type of calculation to perform"
    )
    parser.add_argument(
        "--align",
        default = "name CA",
        help = "Atom selection for alignment (default: name CA)"
    )

    parser.add_argument(
        "--select",
        default = "name CA",
        help = "Atom selection for DCCM/LMI calculation (default: name CA)"
    )

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

    # Aligning the frames w.r.t user input (Default: name CA)
    ref = mda.Universe(args.pdb)    # so it aligns to the topology file
    align.AlignTraj(u, ref, select=args.align, in_memory=True).run()

    atoms = u.select_atoms(args.select)
    coords = []
    for ts in u.trajectory:
        coords.append(atoms.positions.copy())
    coords = np.array(coords)

    fluct = fluct_calc(coords)

    if args.method == "dccm":         # lmi to be added
        result = dccm_calc(fluct)
        output_file = "dccm.dat"

    np.savetxt(output_file, result, delimiter=" ")  # writing to a .dat file

if __name__ == "__main__":
    main()
