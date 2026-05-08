#!/usr/bin/env python

import MDAnalysis as mda
import numpy as np
from MDAnalysis.analysis import align
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
    cov = (X @ X.T) / (n_frames-1) # covariance b/w pairs of residues using matrix multiplication
    var = np.diag(cov)
    denom = np.sqrt(np.outer(var, var))
    denom[denom == 0] = np.nan    # to avoid division by 0 error (if the atom does not move at all)
    dccm = cov / denom
    return dccm

def main():
    parser = argparse.ArgumentParser(description="Compute correlation metrics from MD trajectory")

    parser.add_argument(
        "-s", "--top",
        required = True,
        help = "Topology file"
    )

    parser.add_argument(
        "-t", "--trj",
        required = True,
        help = "Trajectory file"
    )

    parser.add_argument(
        "-m", "--method",
        choices = ["dccm", "lmi"],
        default = "dccm",
        help = "Type of calculation to perform (default: DCCM)"
    )

    parser.add_argument(
        "-a", "--align",
        default = "name CA",
        help = "Atom selection for alignment (default: name CA)"
    )

    parser.add_argument(
        "-sel", "--select",
        default = "name CA",
        help = "Atom selection for DCCM/LMI calculation (default: name CA)"
    )

    parser.add_argument(
        "-o", "--output",
        default = "dccm",
        help = "Output file name (default: dccm.dat)"
    )

    args = parser.parse_args()

    # Loading the structure and trajectory
    try:
        u = mda.Universe(args.top, args.trj, in_memory=True)
    except FileNotFoundError:
        print("Error: One or more files not found")
        sys.exit(1)
    except ValueError:
        print("Error: Invalid file format for pdb or trajectory")
        sys.exit(1)
    except OSError:
        print("Error: One or more files not found")
        sys.exit(1)

    # Aligning the frames w.r.t user input (Default: name CA)
    ref = mda.Universe(args.top)    # Aligning to the reference structure
    align.AlignTraj(u, ref, select=args.align, in_memory=True).run()

    atoms = u.select_atoms(args.select)
    coords = u.trajectory.timeseries(atoms, order="fac")    # so that the shape is (frames, atoms, coords)
    fluct = fluct_calc(coords)

    if args.method == "dccm":         # lmi to be added
        result = dccm_calc(fluct)
        output_file = args.output + ".dat"

    np.savetxt(output_file, result, delimiter=" ")  # writing to a .dat file

    # Writing a CSV file with residue pairs
    csv_output = output_file.replace(".dat", ".csv")

    with open(csv_output, "w") as outfile:
        outfile.write("chain1,residue_number1,residue_name1,atom1,"
                       "chain2,residue_number2,residue_name2,atom2,correlation\n"
        )
        for i in range(len(atoms)):
            for j in range(i+1, len(atoms)):

                outfile.write(
                    f"{atoms[i].chainID},"
                    f"{atoms[i].resid},"
                    f"{atoms[i].resname},"
                    f"{atoms[i].name},"
                    f"{atoms[j].chainID},"
                    f"{atoms[j].resid},"
                    f"{atoms[j].resname},"
                    f"{atoms[j].name},"
                    f"{result[i,j]:.6f}\n"
                )

if __name__ == "__main__":
    main()
