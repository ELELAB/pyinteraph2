#!/usr/bin/env python

import MDAnalysis as mda
import numpy as np
from MDAnalysis.analysis import align
import argparse
import sys

def calculate_dccm(fluct):
    """
    Generates n*n matrix showing the paired movement of the n residues
    according to dynamical cross-correlation
    """
    n_frames, n_res, _ = fluct.shape

    # making the columns into frames and then flattening so that each row is a time series of the motion of each residue
    X = fluct.transpose(1, 0, 2).reshape(n_res, n_frames * 3)
    cov = (X @ X.T) / n_frames # covariance b/w pairs of residues using matrix multiplication
    var = np.diag(cov)
    denom = np.sqrt(np.outer(var, var))
    denom[denom == 0] = np.nan    # to avoid division by 0 error (if the atom does not move at all)
    dccm = cov / denom
    return dccm

def calculate_lmi(fluct):
    """
    Generates n*n matrix showing the paired movement of the n residues
    according to linear mutual information
    """
    n_frames, n_res, _ = fluct.shape
    epsilon = 1e-6

    # computing per residue covariance and log-determinants
    Ci_list = []
    logdet_list = []

    for i in range(n_res):
        Xi = fluct[:,i,:]
        Ci = Xi.T @ Xi / (n_frames - 1)    # calculating 3*3 covariance matrix for each residue
        Ci_list.append(Ci)
        _, logdet = np.linalg.slogdet(Ci + epsilon * np.eye(3))   # log(det(Ci)) and small regularization term added to avoid det(Ci) = 0
        logdet_list.append(logdet)

    # pairwise calculations
    lmi_norm = np.eye(n_res)

    for i in range(n_res):
        Xi = fluct[:,i,:]

        for j in range(i+1, n_res):
            Xj = fluct[:,j,:]

            C_cross = Xi.T @ Xj / (n_frames - 1)   # 3*3 cross-covariance matrix between pairs of residues

            Cij = np.block([                # 6*6 joint covariance matrix where diagonals are covariance matrices of the 2 residues
                [Ci_list[i], C_cross  ],    # and non-diagonals are the cross-covariance matrices
                [C_cross.T, Ci_list[j]]
            ]) + epsilon * np.eye(6)

            _, logdet_Cij = np.linalg.slogdet(Cij)

            raw = max(0.0, 0.5 * (logdet_list[i] + logdet_list[j] - logdet_Cij))
            norm = np.sqrt(1.0 - np.exp(-2.0 * raw/3.0))    # Normalizing

            lmi_norm[i,j] = lmi_norm[j,i] = norm
    return lmi_norm

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
        "-S", "--select",
        default = "name CA",
        help = "Atom selection for DCCM/LMI calculation (default: name CA)"
    )

    parser.add_argument(
        "-o", "--output",
        default = "dccm.dat",
        help = "Output file name for adjacency matrix format (default: dccm.dat)"
    )

    parser.add_argument(
        "-c", "--csv",
        default = "dccm.csv",
        help = "Output file name for CSV format (default: dccm.csv)"
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

    # Align the frames w.r.t user input (Default: name CA)
    ref = mda.Universe(args.top)    # Aligning to the reference structure
    align.AlignTraj(u, ref, select=args.align, in_memory=True).run()

    atoms = u.select_atoms(args.select)
    coords = u.trajectory.timeseries(atoms, order="fac")    # so that the shape is (frames, atoms, coords)

    # Calculate fluctuations
    mean_pos = coords.mean(axis=0)
    fluct = coords - mean_pos

    # Calculate correlation matrices
    if args.method == "dccm":         # lmi to be added
        result = calculate_dccm(fluct)
    elif args.method == "lmi":
        result = calculate_lmi(fluct)

    np.savetxt(args.output, result, delimiter=" ")  # writing to a .dat file

    # Writing a CSV file with residue pairs

    with open(args.csv, "w") as outfile:
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
