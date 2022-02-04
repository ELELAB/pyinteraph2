#!/usr/bin/python
# -*- coding: utf-8 -*-

#    PyInteraph, a software suite to analyze interactions and interaction 
#    network in structural ensembles.
#    Copyright (C) 2013 Matteo Tiberti <matteo.tiberti@gmail.com>, 
#    Gaetano Invernizzi, Yuval Inbar, Matteo Lambrughi, Gideon Schreiber,
#    Elena Papaleo <elena.papaleo@unimib.it> <elena.papaleo@bio.ku.dk>
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.

#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.

#   You should have received a copy of the GNU General Public License
#   along with this program.
#   If not, see <http://www.gnu.org/licenses/>.



# Standard library
import collections
import configparser as cp
import itertools
import json
import logging as log
import os
import re
import struct
import sys
# Third-party packages
import numpy as np
import MDAnalysis as mda
# libinteract
from libinteract import acPSN
from libinteract import innerloops as il



############################## POTENTIAL ##############################



class Sparse:
    def __repr__(self):
        fmt_repr = \
            "<Sparse r1={:d} ({:d},{:d}), r2={:d} ({:d},{:d}), {:d} bins>"
        
        return fmt_repr.format(self.r1, self.p1_1, self.p1_2, self.r2, \
                               self.p2_1, self.p2_2, self.num_bins())
    
    def __init__(self, sparse_list):
        if isinstance(sparse_list, Sparse):
            self.r1 = sparse_list.r1
            self.r1 = sparse_list.r1
            self.r2 = sparse_list.r2
            self.p1_1 = sparse_list.p1_1
            self.p1_2 = sparse_list.p1_2
            self.p2_1 = sparse_list.p2_1
            self.p2_2 = sparse_list.p2_2
            self.cutoff = sparse_list.cutoff
            self.step = sparse_list.step
            self.total = sparse_list.total
            self.num = sparse_list.num
            self.bins = sparse.bins
        
        else:
            self.r1 = sparse_list[0]
            self.r2  = sparse_list[1]
            self.p1_1 = sparse_list[2]
            self.p1_2 = sparse_list[3]
            self.p2_1 = sparse_list[4]
            self.p2_2 = sparse_list[5]
            self.cutoff = np.sqrt(sparse_list[6])
            self.step = 1.0 / sparse_list[7]
            self.total = sparse_list[8]
            self.num = sparse_list[9]
            self.bins  = {}

    def add_bin(self, bin):
        self.bins["".join(bin[0:4])] = bin[4]
   
    def num_bins(self):
        return len(self.bins)


def parse_sparse(potential_file):

    kbp_residues_list = \
        ["ALA","ARG","ASN","ASP","CYS","GLN","GLU","GLY","HIS",
         "ILE","LEU","LYS","MET","PHE","PRO","SER","THR","TRP",
         "TYR","VAL"]
    
    header_fmt  = "400i"
    header_struct = struct.Struct(header_fmt)
    header_size = struct.calcsize(header_fmt)
    sparse_fmt  = "=iiiiiidddidxxxx"
    sparse_size = struct.calcsize(sparse_fmt)
    sparse_struct = struct.Struct(sparse_fmt)
    bin_fmt = "4cf"
    bin_size  = struct.calcsize(bin_fmt)
    bin_struct = struct.Struct(bin_fmt)
    pointer = 0
    sparses = []

    fh = open(potential_file, "rb")
    data = fh.read()
    fh.close()

    isparse = \
        header_struct.unpack(data[pointer : pointer + header_size])
    pointer += header_size

    logstr = "Found {:d} residue-residue interaction definitions."
    log.info(logstr.format(len([i for i in isparse if i > 0])))

    for i in isparse:
        
        sparses.append([])
        
        if i == 0:
            continue
        
        for j in range(i):
            
            this_sparse = \
                Sparse(sparse_struct.unpack(\
                    data[pointer : pointer + sparse_size]))
            pointer += sparse_size
            
            # for every bin ...
            for k in range(this_sparse.num):
                
                tmp_struc = bin_struct.unpack(\
                    data[pointer : pointer + bin_size])
                
                tmp_struc = \
                    tuple([s.decode("ascii") \
                           for s in tmp_struc[0:4] ] + [tmp_struc[4]])
                
                this_sparse.add_bin(tmp_struc)
                pointer += bin_size
            
            sparses[-1].append(this_sparse)
    
    if pointer != len(data):
        errstr = \
            f"Error: could not completely parse the file " \
            f"{potential_file} ({pointer} bytes read, " \
            f"{len(data)} expected)."
        log.error(errstr)
        raise ValueError(errstr)

    sparses_dict = {}
    
    for i in range(len(kbp_residues_list)):
        sparses_dict[kbp_residues_list[i]] = {}
        for j in range(i):            
            sparses_dict[kbp_residues_list[i]][kbp_residues_list[j]] = {}
    
    for s in sparses:
        if s:
            sparses_dict[kbp_residues_list[s[0].r1]][kbp_residues_list[s[0].r2]] = s[0]
   
    logstr = f"Done parsing file {potential_file}!"
    sys.stdout.write(logstr)
    
    return sparses_dict


def parse_atomlist(fname):
    """Parse atom list for potential calculation.
    """

    try:
        fh = open(fname)
    except:
        errstr = f"Could not open file {kbpatomsfile}."
        log.error(errstr)
        raise IOError(errstr)
    
    with fh:
        data = {}
        for line in fh:
            tmp = line.strip().split(":")
            data[tmp[0].strip()] = \
                [i.strip() for i in tmp[1].split(",")]

    return data


def calc_potential(distances,
                   ordered_sparses,
                   kbT = 1.0):
    """Calculate the statistical potential.
    """

    general_cutoff = 5.0  
    tot = 0
    done = 0
    this_dist = np.zeros((2,2), dtype = np.float64)
    scores = np.zeros((distances.shape[1]), dtype = np.float64)
    
    for frame in distances:
        for pn, this_dist in enumerate(frame):
            if np.any(this_dist < general_cutoff):
                searchstring = \
                    "".join(\
                        [chr(int(d * ordered_sparses[pn].step + 1.5)) \
                         for d in this_dist])
                try:
                    probs = \
                        - kbT * ordered_sparses[pn].bins[searchstring]
                
                except KeyError:
                    probs = 0.0
                
                scores[pn] += probs
            
            else:
                scores[pn] += 0.0
    
    return scores/distances.shape[0]


def do_potential(kbp_atomlist,
                 residues_list,
                 potential_file,
                 parse_sparse_func = parse_sparse,
                 calc_potential_func = calc_potential,
                 seq_dist_co = 0,
                 uni = None,
                 pdb = None,
                 do_fullmatrix = True,
                 kbT = 1.0):

    log.info("Loading potential definition...")
    sparses = parse_sparse_func(potential_file)
    log.info("Loading input files...")

    ok_residues = []
    discarded_residues = set()
    residue_pairs = []
    atom_selections = []
    ordered_sparses = []
    numframes = len(uni.trajectory)

    for res in uni.residues:
        
        # Check if the residue type is one of
        # those included in the list
        if res.resname in residues_list:
            # Add it to the accepted residues
            ok_residues.append(res)
        else:
            # Add it to the discarded residues
            discarded_residues.add(res)
            continue
        
        # For each other accepted residue
        for res2 in ok_residues[:-1]:
            res1 = res 
            seq_dist = abs(res1.ix - res2.ix)
            res1_segid = res1.segment.segid
            res2_segid = res2.segment.segid
            
            if not (seq_dist < seq_dist_co or res1_segid != res2_segid):
                
                # String comparison ?!
                if res2.resname < res1.resname:
                    res1, res2 = res2, res1
                
                this_sparse = sparses[res1.resname][res2.resname]
                
                # Get the four atoms for the potential
                atom0, atom1, atom2, atom3 = \
                    (kbp_atomlist[res1.resname][this_sparse.p1_1],
                     kbp_atomlist[res1.resname][this_sparse.p1_2],
                     kbp_atomlist[res2.resname][this_sparse.p2_1],
                     kbp_atomlist[res2.resname][this_sparse.p2_2])
                
                try:
                    index_atom0 = \
                        res1.atoms.names.tolist().index(atom0)
                    index_atom1 = \
                        res1.atoms.names.tolist().index(atom1)
                    index_atom2 = \
                        res2.atoms.names.tolist().index(atom2)
                    index_atom3 = \
                        res2.atoms.names.tolist().index(atom3)
                    selected_atoms = \
                        mda.core.groups.AtomGroup((\
                             res1.atoms[index_atom0],
                             res1.atoms[index_atom1],
                             res2.atoms[index_atom2],
                             res2.atoms[index_atom3]))
                except:
                    # Inform the user about the problem and
                    # continue
                    warnstr = \
                        f"Could not identify essential atoms for " \
                        f"the analysis ({res1.resname}{res1.resid}," \
                        f" {res2.resname}{res2.resid})"
                    log.warning(warnstr)
                    continue
                
                residue_pairs.append((res1, res2))
                atom_selections.append(selected_atoms)
                ordered_sparses.append(this_sparse)

    # Create an matrix of floats to store scores (initially
    # filled with zeros)
    scores = np.zeros((len(residue_pairs)), dtype = np.float64)
    
    # Set coordinates to None
    coords = None
    
    # For each frame in the trajectory
    numframe = 1
    
    for ts_i, ts in enumerate(uni.trajectory):
        
        # Log the progress along the trajectory
        logstr = "Now analyzing: frame {:d} / {:d} ({:3.1f}%)\r"
        sys.stdout.write(\
            logstr.format(ts_i + 1,
                          numframes,
                          float(numframe)/float(numframes)*100.0))
        sys.stdout.flush()   
        
        # Create an array of coordinates by concatenating the arrays of
        # atom positions in the selections row-wise
        coords = \
            np.array(\
                np.concatenate(\
                    [sel.positions for sel in atom_selections]),
            dtype = np.float64)

        inner_loop = il.LoopDistances(coords, 
                                      coords,
                                      None,
                                      corrections = None)
        
        # Compute distances
        distances = \
            inner_loop.run_potential_distances(len(atom_selections),
                                               4,
                                               1)
        
        # Compute scores
        scores += \
            calc_potential_func(distances = distances,
                                ordered_sparses = ordered_sparses,
                                kbT = kbT)

        # Update the frame number
        numframe += 1
    
    # Divide the scores for the lenght of the trajectory
    scores /= float(numframes)

    # Create an empty list to store the final table
    table = []
    
    for i, score in enumerate(scores):
        if abs(score) > 0.000001:
            
            # Update the output table
            table.append(\
                 (pdb.residues[residue_pairs[i][0].ix].segment.segid,
                  pdb.residues[residue_pairs[i][0].ix].resnum,
                  pdb.residues[residue_pairs[i][0].ix].resname,
                  "sidechain",
                  pdb.residues[residue_pairs[i][1].ix].segment.segid,
                  pdb.residues[residue_pairs[i][1].ix].resnum,
                  pdb.residues[residue_pairs[i][1].ix].resname,
                  "sidechain",
                  score))
    
    # Inizialize the matrix to None  
    dm = None   
    
    if do_fullmatrix:
        
        # If requested, create the matrix
        dm = np.zeros((len(pdb.residues), len(pdb.residues)))
        
        # Use numpy "fancy indexing" to fill the matrix
        # with scores at the corresponding residues pairs
        # positions
        pair_firstelems = [pair[0].ix for pair in residue_pairs]
        pairs_secondelems = [pair[1].ix for pair in residue_pairs]
        dm[pair_firstelems, pairs_secondelems] = scores
        dm[pairs_secondelems, pair_firstelems] = scores
    
    # Return the output table and the matrix
    return (table, dm)



############################ INTERACTIONS #############################



def parse_cgs_file(fname):
    """Parse the file contaning the charged groups to be
    used for the calculation of electrostatic interactions.
    """
    
    grps_str = "CHARGED_GROUPS"
    res_str = "RESIDUES"
    default_grps_str = "default_charged_groups"

    cfg = cp.ConfigParser()
    
    try:
        cfg.read(fname)
    except Exception as e:
        errstr = \
            f"File {fname} not readable or not in the right format." \
            f"Exception: {e}"
        raise IOError(errstr)

    out = {}
    group_definitions = {}

    charged_groups = cfg.options(grps_str)
    charged_groups.remove(default_grps_str)
    charged_groups = [i.strip() for i in charged_groups]

    default_charged = cfg.get(grps_str, default_grps_str).split(",")
    default_charged = [i.strip() for i in default_charged]

    residues = cfg.options(res_str)

    for i in charged_groups + default_charged:
        group_definitions[i] = \
            [s.strip() for s in cfg.get(grps_str, i).split(",")]
    
    for j in range(len(group_definitions[i])):
        group_definitions[i][j] = \
            group_definitions[i][j].strip()

    try:
        for i in residues:
            i = i.upper()
            out[i] = {}
            for j in default_charged:
                out[i][j] = group_definitions[j]
            this_cgs = [s.strip() for s in cfg.get(res_str, i).split(",")]
            for j in this_cgs:
                if j:
                    out[i][j] = group_definitions[j.lower()]
    except Exception as e:
        errstr = \
            f"Could not parse the charged groups file {fname}. " \
            f"Are there any inconsistencies? Exception: {e}"
        raise ValueError(errstr)

    return out


def null_correction(chosenselections, frame):
    """Do not incorporate correction factors into
    the persistence calculation.
    """
    
    return np.repeat(0.0, len(chosenselections))


def rg_correction(chosenselections, frame):
    """Compute rg correction factor for the given
    residue at the given frame."""
    
    return np.array([sel.radius_of_gyration() \
                    for sel in chosenselections], dtype = np.float64)


def calc_dist_matrix(uni,
                     idxs,
                     chosenselections,
                     co,
                     mindist = False,
                     mindist_mode = None,
                     pos_char = "p",
                     neg_char = "n",
                     correction_func = null_correction):
    """Compute matrix of distances.
    """

    # Get the number of frames in the trajectory
    numframes = len(uni.trajectory)
    
    # Initialize the final matrix
    percmat = \
        np.zeros((len(chosenselections), len(chosenselections)),
                 dtype = np.float64)

    # If electrostatic contacts calculation has been requested
    if mindist:
        
        # Lists for positively charged atoms
        pos = []
        pos_idxs = []
        pos_sizes = []
        
        # Lists for negatively charged atoms
        neg = []
        neg_idxs = []
        neg_sizes = []

        for i in range(len(idxs)):
            
            # Character in the index indicating the
            # charge of the atom
            charge_char = idxs[i][3][-1]
            
            # If the index contains the indication of a
            # positively charged atom
            if charge_char == pos_char:
                pos.append(chosenselections[i])
                pos_idxs.append(idxs[i])
                pos_sizes.append(len(chosenselections[i]))
            
            # If the index contains the indication of a
            # negatively charged atom
            elif charge_char == neg_char:
                neg.append(chosenselections[i])
                neg_idxs.append(idxs[i])
                neg_sizes.append(len(chosenselections[i]))
            
            # If none of the above is true, raise an exception
            else:
                errstr = \
                    f"Accepted values are '{pos_char}' and " \
                    f"'{neg_char}', but {charge_char} was found."
                raise ValueError(errstr)

        # Convert the lists of atom positions into arrays
        pos_sizes = np.array(pos_sizes, dtype = np.int)
        neg_sizes = np.array(neg_sizes, dtype = np.int)

        # If we are interested in interactions between atoms
        # with different charges
        if mindist_mode == "diff":
            sets = [(pos, neg)]
            sets_idxs = [(pos_idxs, neg_idxs)]
            sets_sizes = [(pos_sizes, neg_sizes)]
        
        # If we are interested in interactions between atoms
        # with the same charge
        elif mindist_mode == "same":
            sets = [(pos, pos), (neg, neg)]
            sets_idxs = [(pos_idxs, pos_idxs), (neg_idxs, neg_idxs)]
            sets_sizes = [(pos_sizes, pos_sizes), (neg_idxs, neg_sizes)]
        
        # If we are interested in both
        elif mindist_mode == "both":
            sets = [(chosenselections, chosenselections)]
            sets_idxs = [(idxs, idxs)]
            sizes =  [len(s) for s in chosenselections]
            sets_sizes = [(sizes, sizes)]
        
        # Unrecognized choice
        else:
            choices = ["diff", "same", "both"]
            errstr = \
                f"Accepted values for 'mindist_mode' are " \
                f"{', '.join(choices)}, but {mindist_mode} was found."
            raise ValueError(errstr)

        # Create an empty list to store the contacts 
        percmats = []
        
        # Create an empty list to store the atomic coordinates
        coords = [([[], []]) for s in sets]

        # Set the initial frame number to 1
        numframe = 1

        # For each frame in the trajectory
        for ts in uni.trajectory:
            
            # Log the progress along the trajectory
            logstr = \
                "Caching coordinates: frame {:d} / {:d} ({:3.1f}%)\r"
            sys.stdout.write(logstr.format(\
                                numframe, \
                                numframes, \
                                float(numframe)/float(numframes)*100.0))
            sys.stdout.flush()
            
            # Update the frame number
            numframe += 1

            # For each set of atoms
            for s_index, s in enumerate(sets):

                # Triangular case               
                if s[0] == s[1]:
                    log.info("Caching coordinates...")
                    for group in s[0]:
                        coords[s_index][0].append(group.positions)
                        coords[s_index][1].append(group.positions)
                
                # Square case
                else:
                    log.info("Caching coordinates...")
                    for group in s[0]:
                        coords[s_index][0].append(group.positions)
                    for group in s[1]:
                        coords[s_index][1].append(group.positions)

        for s_index, s in enumerate(sets):
            
            # Triangular case
            if s[0] == s[1]:
                
                # Get the coordinates
                this_coords = \
                    np.array(np.concatenate(coords[s_index][0]),
                             dtype = np.float64)

                # Compute the distances within the cut-off
                inner_loop = il.LoopDistances(this_coords,
                                              this_coords,
                                              co,
                                              corrections = None)
                
                # Update the final matrix
                percmats.append(\
                    inner_loop.run_triangular_mindist(\
                        sets_sizes[s_index][0]))

            # Square case
            else:

                # Get the coordinates of the first set of atoms
                this_coords1 = \
                    np.array(np.concatenate(coords[s_index][0]),
                             dtype = np.float64)
                
                # Get the coordinates of the second set of atoms
                this_coords2 = \
                    np.array(np.concatenate(coords[s_index][1]),
                             dtype = np.float64)
                
                # Compute the distances within the cut-off
                inner_loop = il.LoopDistances(this_coords1,
                                              this_coords2,
                                              co,
                                              corrections = None)

                # Update the final matrix
                percmats.append(\
                    inner_loop.run_square_mindist(\
                        sets_sizes[s_index][0],
                        sets_sizes[s_index][1]))

        for s_index, s in enumerate(sets):
            
            # Recover the final matrix
            pos_idxs = sets_idxs[s_index][0]
            neg_idxs = sets_idxs[s_index][1]

            # Triangular case
            if s[0] == s[1]:
                for j in range(len(s[0])):
                    for k in range(0, j):
                        ix_j = idxs.index(pos_idxs[j])
                        ix_k = idxs.index(pos_idxs[k])
                        percmat[ix_j, ix_k] = percmats[s_index][j,k]
                        percmat[ix_k, ix_j] = percmats[s_index][j,k]

            # Square case
            else:
                for j in range(len(s[0])):
                    for k in range(len(s[1])):
                        ix_j_p = idxs.index(pos_idxs[j])
                        ix_k_n = idxs.index(neg_idxs[k])
                        percmat[ix_j_p, ix_k_n] = percmats[s_index][j,k]
                        percmat[ix_k_n, ix_j_p] = percmats[s_index][j,k]

    # If hydrophobic contacts' calculation was requested
    else:

        # Create an empty list of matrices of centers of mass
        all_coms = []

        # Create an empty list of corrections
        all_corrections = []

        # Set the initial frame number to 1
        numframe = 1

        # For each frame in the trajectory
        for ts_i, ts in enumerate(uni.trajectory):
            
            # Log the progress along the trajectory
            logstr = "Now analyzing: frame {:d} / {:d} ({:3.1f}%)\r"
            sys.stdout.write(logstr.format(\
                                numframe,
                                numframes,
                                float(numframe)/float(numframes)*100.0))
            sys.stdout.flush()
            
            # Update the frame number
            numframe += 1

            # List representing the matrix of centers of mass
            # for the chosen selections
            coms_list = [sel.center(sel.masses) for sel in chosenselections]

            # Array representing the matrix of correction factors
            # for the chosen selections
            corrections = correction_func(chosenselections, ts_i)

            # Convert the list of centers of mass into an array
            coms = np.array(coms_list, dtype = np.float64)

            # Append the array of centers of mass to the list
            # containing centers of mass for all frames
            all_coms.append(coms)

            # Append the array of corrections to the list
            # containing corrections for all frames
            all_corrections.append(corrections)

        # Create a matrix of all centers of mass
        # along the trajectory
        all_coms = np.concatenate(all_coms)

        # Create a matrix of all correction factors
        # along the trajectory
        all_corrections = np.concatenate(all_corrections)

        # Compute the distances within the cut-off
        inner_loop = il.LoopDistances(all_coms,
                                      all_coms, 
                                      co, 
                                      corrections = all_corrections)
        percmat = inner_loop.run_triangular_distmatrix(coms.shape[0])

    # Convert the final matrix into an array
    percmat = np.array(percmat, dtype = np.float64)/numframes*100.0

    # Return the final matrix
    return percmat


def assign_ff_masses(ffmasses, chosenselections):
    """Assign atomic masses to atoms in selections.
    """

    # Load force field data
    with open(ffmasses, 'r') as fh:
        ffdata = json.load(fh)

    # For each atom selection
    for selection in chosenselections:
        
        # For each atom in the selection
        for atom in selection:
            
            atom_resname = atom.residue.resname
            atom_resid = atom.residue.resid
            atom_name = atom.name
            
            # Try to assign the mass to the atom
            try:
                atom.mass = ffdata[1][atom_resname][atom_name]

            # In case something went wrong, warn the user
            except:         
                warnstr = \
                    f"Atom type not recognized (resid {atom_resid}, " \
                    f"resname {atom_resname}, atom {atom_name}). " \
                    f"Atomic mass will be guessed."         
                log.warning(warnstr)


def generate_cg_identifiers(pdb, uni, **kwargs):
    """Generate charged atoms identifiers for the
    calculation of electrostatic interactions.
    """

    # Get the charged atoms
    cgs = kwargs["cgs"]
    
    # Preprocess charged groups
    filter_func_p = lambda x: not x.startswith("!")
    filter_func_n = lambda x:     x.startswith("!")   

    # For each residue and the dictionary of charged groups
    # associated to it
    for res, dic in cgs.items():

        # For each charged group
        for cgname, cg in dic.items():
            
            # True : atoms that must exist (negative of negative)
            true_set = set(filter(filter_func_p, cg))
            # False: atoms that must NOT exist (positive of negative)
            false_set = set([j[1:] for j in filter(filter_func_n, cg)])
            # Update the charged groups
            cgs[res][cgname] =  {True : true_set, False : false_set}
    
    # Create a list of identifiers
    identifiers = \
        [(r.segid, r.resid, r.resname, "") for r in pdb.residues]
    
    # Create empty lists for IDs and atom selections
    idxs = []
    chosenselections = []
    
    # Atom selection string
    selstring = "segid {:s} and resid {:d} and (name {:s})"
    
    # For each residue in the Universe
    for k, res in enumerate(uni.residues):
        
        segid = res.segid
        resname = res.resname
        resid = res.resid
        
        # Try to get the charged groups associated to the residue
        try:
            cgs_items = cgs[resname].items()

        # If something went wrong, warn the user and continue
        except KeyError:
            logstr = \
                f"Residue {resname} is not in the charge " \
                f"recognition set. Will be skipped."
            log.warning(logstr)
            continue
        
        # Get the names of the current atoms
        setcurnames = set(res.atoms.names)
        
        # For each charged group associated to the residue
        for cgname, cg in cgs_items:
            
            # Get the set of atoms to be kept
            atoms_must_exist = cg[True]
            atoms_must_not_exist = cg[False]
            
            # Set the condition to keep atoms in the current
            # residue, i.e. the atoms that must be present are
            # present and those which must not be present are not

            condition_to_keep = \
                atoms_must_exist.issubset(setcurnames) and \
                atoms_must_not_exist.isdisjoint(setcurnames)
            
            # If the condition is met
            if condition_to_keep:
                
                # Create the index
                idx = (pdb.residues[k].segment.segid,
                       pdb.residues[k].resid,
                       pdb.residues[k].resname,
                       cgname)
                
                # Append the index to the list of indexes
                idxs.append(idx)

                # Atom selection string for the atoms that must exist
                atoms_str = " or name ".join(atoms_must_exist)

                # Create the atom selection
                selection = \
                    uni.select_atoms(selstring.format(\
                        segid, resid, atoms_str))                
                
                # Update the lists of IDs and atom selections
                chosenselections.append(selection)
                
                # Log the selection
                atoms_names_str = ", ".join([a.name for a in selection])
                logstr = f"{resid} {resname} ({atoms_names_str})"
                log.info(logstr)

    # return identifiers, IDs and atom selections
    return (identifiers, idxs, chosenselections)


def generate_sc_identifiers(pdb, uni, **kwargs):
    """Generate side chain identifiers for the calculation
    of side centers of mass' contacts.
    """

    # Get the residue names list
    reslist = kwargs["reslist"]
    
    # Log the list of residue names
    logstr = f"Selecting residues: {', '.join(reslist)}"
    log.info(logstr)
    
    # Backbone atoms must be excluded
    backbone_atoms = \
        ["CA", "C", "O", "N", "H", "H1", "H2", \
         "H3", "O1", "O2", "OXT", "OT1", "OT2"]
    
    # Create the list of identifiers
    identifiers = \
        [(r.segid, r.resid, r.resname, "sidechain") \
         for r in pdb.residues]
    
    # Create empty lists for IDs and atom selections
    chosenselections = []
    idxs = []
    
    # Atom selection string
    selstring = "segid {:s} and resid {:d} and (name {:s})"
    
    # Start logging the chosen selections
    log.info("Chosen selections:")
    
    # For each residue name in the residue list
    for resname_in_list in reslist:

        # Get the first three letters of the residue name
        resname_3letters = resname_in_list[0:3]
        
        # Update the list of IDs with all those residues
        # matching the current residue type
        for identifier in identifiers:
            if identifier[2] == resname_3letters:
                idxs.append(identifier)
        
        # For each residue in the Universe
        for res in uni.residues:

            if res.resname[0:3] == resname_in_list:

                resid = res.resid
                segid = res.segid
                resname = res.resname
                
                # Get only side chain atom names
                sc_atoms_names = \
                    [a.name for a in res.atoms \
                     if a.name not in backbone_atoms]
                
                # String for selecting side chain atoms
                sc_atoms_str = " or name ".join(sc_atoms_names)

                # String for logging the atom selection
                sc_atoms_names_str = ", ".join(sc_atoms_names)
                
                # Get the side chain atom selection
                selection = \
                    uni.select_atoms(selstring.format(\
                        segid, resid, sc_atoms_str))
                
                # Append the selection to the list of
                # atom selections
                chosenselections.append(selection)
                
                # Log the selection
                logstr = f"{resid} {resname} ({sc_atoms_names_str})"
                log.info(logstr)

    # Return identifiers, indexes and chosen selections
    return (identifiers, idxs, chosenselections)


def calc_sc_fullmatrix(identifiers, idxs, percmat, perco):
    """Calculate side chain-side chain interaction matrix
    (hydrophobic contacts).
    """
    
    # Create a matrix of size identifiers x identifiers
    fullmatrix = np.zeros((len(identifiers), len(identifiers)))
    
    # Get where (index) the elements of idxs are in identifiers
    where_idxs_in_identifiers = \
        [identifiers.index(item) for item in idxs]
    
    # Get where (index) each element of idxs is in idxs
    where_idxs_in_idxs = [i for i, item in enumerate(idxs)]
    
    # Get where (i,j coordinates) each element of idxs is in
    # fullmatrix
    positions_identifiers_in_fullmatrix = \
        itertools.combinations(where_idxs_in_identifiers, 2)
    
    # Get where (i,j coordinates) each element of idxs is in
    # percmat (which has dimensions len(idxs) x len(idxs))
    positions_idxs_in_percmat = \
        itertools.combinations(where_idxs_in_idxs, 2)
    
    # Unpack all pairs of i,j coordinates in lists of i 
    # indexes and j indexes
    i_fullmatrix, j_fullmatrix = \
        zip(*positions_identifiers_in_fullmatrix)
    i_percmat, j_percmat = \
        zip(*positions_idxs_in_percmat)
    
    # Use numpy "fancy indexing" to fill fullmatrix with the
    # values in percmat corresponding to each pair of elements
    fullmatrix[i_fullmatrix, j_fullmatrix] = \
        percmat[i_percmat,j_percmat]
    fullmatrix[j_fullmatrix, i_fullmatrix] = \
        percmat[i_percmat,j_percmat]
    
    # Return the full matrix (square matrix)
    return fullmatrix


def calc_cg_fullmatrix(identifiers, idxs, percmat, perco):
    """Calculate charged atoms interaction matrix (electrostatic
    interactions).
    """
    
    # Search for residues with duplicate ID
    duplicates = []
    idx_index = 0
    while idx_index < percmat.shape[0]:
        # Function to retrieve only residues whose ID (segment ID,
        # residue ID and residue name) perfectly matches the one of
        # the residue currently under evaluation (ID duplication)
        filter_func = lambda x: x[0:3] == idxs[idx_index][0:3]
        # Get where (indexes) residues with the same ID as that
        # currently evaluated are
        rescgs = list(map(idxs.index, filter(filter_func, idxs)))
        # Save the indexes of the duplicate residues
        duplicates.append(frozenset(rescgs))
        # Update the counter
        idx_index += len(rescgs)
    
    # If no duplicates are found, the corrected matrix will have
    # the same size as the original matrix 
    corrected_percmat = np.zeros((len(duplicates), len(duplicates)))
    
    # Generate all the possible combinations of the sets of 
    # duplicates (each set represents a residue who found
    # multiple times in percmat)
    dup_combinations = itertools.combinations(duplicates, 2)
    
    # For each pair of sets of duplicates
    for dup_resi, dup_resj in dup_combinations:
        # Get where residue i and residue j should be uniquely
        # represented in the corrected matrix
        corrected_ix_i = duplicates.index(dup_resi)
        corrected_ix_j = duplicates.index(dup_resj)
        # Get the positions of all interactions made by each instance
        # of residue i with each instance of residue j in percmat
        index_i, index_j = zip(*itertools.product(dup_resi,dup_resj))
        # Use numpy "fancy indexing" to put in corrected_percmat
        # only the strongest interaction found between instances of
        # residue i and instances of residue j
        corrected_percmat[corrected_ix_i,corrected_ix_j] = \
            np.max(percmat[index_i,index_j])
    
    # To generate the new IDs, get the first instance of each residue
    # (duplicates all share the same ID) and add an empty string as
    # last element of the ID
    corrected_idxs = \
        [idxs[list(i)[0]][0:3] + ("",) for i in list(duplicates)]
    
    # Create a matrix of size identifiers x identifiers
    fullmatrix = np.zeros((len(identifiers), len(identifiers)))
    
    # Get where (index) the elements of corrected_idxs are in identifiers
    where_idxs_in_identifiers = \
        [identifiers.index(item) for item in corrected_idxs]
    
    # Get where (index) each element of corrected_idxs 
    # is in corrected_idxs
    where_idxs_in_idxs = [i for i, item in enumerate(corrected_idxs)]
    
    # Get where (i,j coordinates) each element of corrected_idxs
    # is in fullmatrix
    positions_identifiers_in_fullmatrix = \
        itertools.combinations(where_idxs_in_identifiers, 2)
    
    # Get where (i,j coordinates) each element of corrected_idxs
    # is in corrected_percmat
    positions_idxs_in_corrected_percmat = \
        itertools.combinations(where_idxs_in_idxs, 2)
    
    # Unpack all pairs of i,j coordinates in lists of i 
    # indexes and j indexes
    i_fullmatrix, j_fullmatrix = \
        zip(*positions_identifiers_in_fullmatrix)
    i_corrected_percmat, j_corrected_percmat = \
        zip(*positions_idxs_in_corrected_percmat)
    
    # Use numpy "fancy indexing" to fill fullmatrix with the
    # values in percmat corresponding to each pair of elements
    fullmatrix[i_fullmatrix, j_fullmatrix] = \
        corrected_percmat[i_corrected_percmat, j_corrected_percmat]
    fullmatrix[j_fullmatrix, i_fullmatrix] = \
        corrected_percmat[i_corrected_percmat, j_corrected_percmat]
    
    # Return the full matrix (square matrix)
    return fullmatrix


def filter_by_chain(chain1, chain2, table):
    """Takes in a table, two chain names and the column number of the 
    second chain. Filters the table to only keep rows where the first 
    column contains chain1 and the second column contains chain2. 
    Returns filtered table.
    """

    # Get the rows and columns from the table
    rows = table
    cols = table.T
    
    # If the chains are the same, find all rows where both columns
    # match the chain
    if chain1 == chain2:
        logical_vector = np.logical_and(chain1 == cols[0],
                                        chain2 == cols[4])

    # If the chains are different, check both directions
    else:
        # Check forward direction (e.g. (A, B))
        logical_vector1 = np.logical_and(chain1 == cols[0],
                                         chain2 == cols[4])
        # Check backward direction (e.g. (B, A))
        logical_vector2 = np.logical_and(chain2 == cols[0],
                                         chain1 == cols[4])
        # Combine the vectors
        logical_vector = np.logical_or(logical_vector1, logical_vector2)
    
    # Filter rows
    filtered_rows = rows[logical_vector]
    
    # Warn if no contacts found and return None
    if filtered_rows.shape[0] == 0:
        if chain1 == chain2:
            warnstr = f"No intrachain contacts found in chain {chain1}."
            log.warning(warnstr)
        else:
            warnstr = f"No interchain contacts found between chains " \
                      f"{chain1} and {chain2}."
            log.warning(warnstr)
        return None

    # Otherwise, return the filtered rows
    else:
        return filtered_rows


def create_dict_tables(table):
    """Takes in a single table (list of tuples) and returns a dictionary
    of tables (arrays). Each key in this dictionary represent whether
    the table contains all/intrachain/interchain contacts.
    """

    # Convert to array and keep transpose for selection
    table_rows = np.array(table)
    table_cols = table_rows.T
    
    # Initialize output dictionary of tables
    dict_tables = {"all" : table_rows}
    
    # Find unique chains in the table
    chains = np.unique(np.concatenate((table_cols[0], table_cols[4])))
    
    # If multiple chains are present, split the contacts by chain
    if len(chains) > 1:
	   
       # For each chain, add the nodes that are only in contact
       # with the same chain
        for chain in chains:
            filtered_rows_intra = filter_by_chain(chain,
                                                  chain,
                                                  table_rows)
            # Check that the rows exist
            if filtered_rows_intra is not None:
                dict_tables[chain] = filtered_rows_intra
        
        # Create a vector of all nodes that are in contact
        # with different chains
        logical_vector = table_cols[0] != table_cols[4]
        
        # Warn if no interchain contacts found
        if logical_vector.sum() == 0:
            log.warning("No interchain contacts found.")
        
        # Otherwise, find intrachain contacts for each pair of chains
        else:
            
            # For all combinations of different chains, find nodes
            # that are in contact with different chains
            for chain1, chain2 in itertools.combinations(chains, 2):  
                filtered_rows_inter = filter_by_chain(chain1,
                                                      chain2,
                                                      table_rows)
                
                # Check that the rows exist
                if filtered_rows_inter is not None:
                    name = tuple(sorted([chain1, chain2]))
                    dict_tables[name] = filtered_rows_inter
    
    # Return the dictionary
    return dict_tables


def create_dict_matrices(fullmatrix, dict_tables, pdb):
    """Takes in the full matrix of persistence values and a dictionary
    of tables where each key represents all/intrachain/interchain
    contacts and returns a dictionary of matrices for each key.
    """

    # Initialize output dictionary of matrices
    dict_matrices = {"all": fullmatrix}
    
    # Get chain ID
    res_chain = pdb.residues.segids
    
    # Get residue number
    res_num = pdb.residues.resids
    
    # Map the residue ID (e.g. A4) to its matrix index
    res_dict = \
        {res_chain[i]+str(res_num[i]):i \
         for i in range(fullmatrix.shape[0])}
    
    # Create a matrix for each table and store it with the same key
    for key, element in dict_tables.items():
        
        # Exclude the all chains table
        if key != "all":
            # Create empty matrix
            matrix = np.zeros(fullmatrix.shape)
            # Get list of indexes for the matrix
            table_cols = element.T
            # Combine table cols to get dict key and check dict for index
            mat_i = [res_dict[res] for res in np.char.add(table_cols[0], 
                                                          table_cols[1])]
            mat_j = [res_dict[res] for res in np.char.add(table_cols[4], 
                                                          table_cols[5])]
            # Fill the matrix with appropriate values
            matrix[mat_i, mat_j] = fullmatrix[mat_i, mat_j]
            matrix[mat_j, mat_i] = fullmatrix[mat_j, mat_i]
            # Insert matrix into dictionary
            dict_matrices[key] = matrix
    
    # Return the dictionary of matrices
    return dict_matrices


def save_output_dict(out_dict, filename):
    """Save each value in a dictionary as a separate file. Must specify
    if the dictionary is a CSV file or matrix. Saves a CSV by default.
    """

    # Remove the extension from the file name if present
    filename = os.path.splitext(filename)[0]
    
    # Check if the table or the matrix is being saved
    # and change the parameters
    if out_dict["all"].dtype == float:
        ext = ".dat"
        delim = ' '
        _format = "%.1f"
    else:
        ext = ".csv"
        delim = ','
        _format = '%s'
    
    # For each key, change the filename and save the dict value
    for key in out_dict:
        if key == "all":
            fname = f"{filename}_all{ext}"
        # Check if only one chain is present (intrachain)
        elif len(key) == 1:
            fname = f"{filename}_intra_{key}{ext}"
        # Check if there are interchain contacts
        elif len(key) == 2:
            fname = f"{filename}_inter_{key[0]}-{key[1]}{ext}"
        # Save the file
        np.savetxt(fname,
                   out_dict[key],
                   delimiter = delim,
                   fmt = _format)


def do_interact(pdb,
                uni,
                identfunc = None,
                co = 5.0,
                perco = 0.0,
                assignffmassesfunc = assign_ff_masses,
                distmatrixfunc = calc_dist_matrix,
                ffmasses = None,
                fullmatrixfunc = None,
                mindist = False,
                mindist_mode = None,
                correction_func = None,
                **identargs):
    """Calculate either electrostatic interactions or side
    chain centers' of mass contacts.
    """
    
    # If not correction function is provided, no correction will
    # be applied
    if correction_func is None:
        correction_func = null_correction

    # Get identifiers, indexes and atom selections
    identifiers, idxs, chosenselections = \
        identfunc(pdb, uni, **identargs)

    # Assign atomic masses to atomic selections if not provided
    if ffmasses is None:
        log.info("No force field assigned: masses will be guessed.")
    else:
        try:
            assignffmassesfunc(ffmasses, chosenselections)
        except IOError:
            logstr = "Force field file not found or not readable. " \
                     "Masses will be guessed."
            log.warning(logstr)     
    
    # Calculate the matrix of persistences
    percmat = calc_dist_matrix(uni = uni,
                               idxs = idxs,
                               chosenselections = chosenselections,
                               co = co,
                               mindist = mindist,
                               mindist_mode = mindist_mode,
                               correction_func = correction_func)

    # Create empty list for output
    table = []
    
    # Get where in the lower triangle of the matrix (it is symmeric)
    # the value is greater than the persistence cut-off
    where_gt_perco = np.argwhere(np.tril(percmat>perco))
    for i, j in where_gt_perco:
        res1 = idxs[i]
        res2 = idxs[j]
        persistence = (percmat[i,j],)
        table.append(res1 + res2 + persistence)
    
    # Set the full matrix to None
    fullmatrix = None
    
    # Compute the full matrix if requested
    if fullmatrixfunc is not None:
        fullmatrix = fullmatrixfunc(identifiers = identifiers,
                                    idxs = idxs,
                                    percmat = percmat,
                                    perco = perco)
    
    # Return the output list and full matrix
    return table, fullmatrix



############################### acPSN #################################



def parse_nf_file(fname):
    """Parse the file contaning the normalization factors to be
    used for the calculation of the acPSN.
    """
    
    # String indicating the section to get the normalization
    # factors from
    nf_str = "NORMALIZATION_FACTORS"

    # Create the configparser
    cfg = cp.ConfigParser()
    
    # Try to read the file
    try:
        cfg.read(fname)
    # If something went wrong, log the error and exit 
    except Exception as e:
        errstr = \
            f"File {fname} not readable or not in the right format. " \
            f"Exception : {e}"
        log.error(errstr)
        raise IOError(errstr)

    # Try to parse the file
    try:
        nf = {r.upper() : float(cfg.get(nf_str, r)) \
              for r in cfg.options(nf_str)}
    # If something went wrong, log the error and exit
    except Exception as e:
        errstr = \
            f"Could not parse the normalization factors file " \
            f"{fname}. Are there any inconsistencies? Exception: {e}"
        log.error(errstr)
        raise ValueError(errstr)

    # Return the dictionary of normalization factors
    return nf


def do_acpsn(pdb,
             uni,
             co,
             perco,
             proxco,
             imin,
             edge_weight,
             norm_facts,
             nf_permissive,
             nf_default):
    """Compute the atomic contacts-based PSN as devised by
    Vishveshvara's group.
    """

    # Create the acPSN constructor
    builder = acPSN.AtomicContactsPSNBuilder()
    # Return the table of contacts and the acPSN
    return builder.get_average_psn(universe = uni,
                                   universe_ref = pdb,
                                   i_min = imin,
                                   dist_cut = co,
                                   prox_cut = proxco,
                                   p_min = perco,
                                   norm_facts = norm_facts,
                                   edge_weight = edge_weight,
                                   permissive = nf_permissive,
                                   norm_fact_default = nf_default)



############################### HBONDS ################################



def parse_hbs_file(fname):
    """Parse the file containing the groups making hydrogen
    bonds.
    """

    hbs_str = "HYDROGEN_BONDS"
    acceptors_str = "ACCEPTORS"
    donors_str = "DONORS"
    
    cfg = cp.ConfigParser()

    try:
        cfg.read(fname)
    except:
        logstr = \
            f"File {fname} not readeable or not in the right format."
        log.error(logstr)

    acceptors = cfg.get(hbs_str, acceptors_str)
    acceptors = [i.strip() for i in acceptors.strip().split(",")]

    donors = cfg.get(hbs_str, donors_str)
    donors = [i.strip() for i in donors.strip().split(",")]

    return dict(ACCEPTORS = acceptors, DONORS = donors)


def do_hbonds(sel1,
              sel2,
              pdb,
              uni,
              update_selection1 = True,
              update_selection2 = True,
              filter_first = False,
              distance = 3.0,
              angle = 120,
              perco = 0.0,
              perresidue = False,
              do_fullmatrix = False,
              other_hbs = None):
    
    # Import the hydrogen bonds analysis module
    from MDAnalysis.analysis.hbonds import hbond_analysis
    
    # Check if selection 1 is valid
    try:
        sel1atoms = uni.select_atoms(sel1)
    except:
        log.error("ERROR: selection 1 is invalid")
    
    # Check if selection 2 is valid
    try:
        sel2atoms = uni.select_atoms(sel2)
    except:
        log.error("ERROR: selection 2 is invalid")      
    
    # Check if custom donors and acceptors were provided
    if other_hbs is None:
        class Custom_HydrogenBondAnalysis(hbond_analysis.HydrogenBondAnalysis):
            pass
        hb_ff = "CHARMM27"
    else:  
        # Custom names
        class Custom_HydrogenBondAnalysis(hbond_analysis.HydrogenBondAnalysis):
            DEFAULT_DONORS = {"customFF" : other_hbs["DONORS"]}
            DEFAULT_ACCEPTORS = {"customFF" : other_hbs["ACCEPTORS"]}
        hb_ff = "customFF"
    
    # Set up the hydrogen bonds analysis
    h = Custom_HydrogenBondAnalysis(universe = uni,
                                    selection1 = sel1,
                                    selection2 = sel2,
                                    distance = distance,
                                    angle = angle,
                                    forcefield = hb_ff,
                                    update_selection1 = update_selection1,
                                    update_selection2 = update_selection2,
                                    filter_first = filter_first)
    
    # Inform the user about the hydrogen bond analysis parameters
    logstr = "Will use {:s}: {:s}"
    log.info(logstr.format("acceptors", \
                           ", ".join(h.DEFAULT_ACCEPTORS[hb_ff])))
    log.info(logstr.format("donors", \
                           ", ".join(h.DEFAULT_DONORS[hb_ff])))
    log.info("Running hydrogen bonds analysis...")
    
    # Run the hydrogen bonds analysis
    h.run()
    log.info("Done! Finalizing...")
    
    # Get the hydrogen bonds timeseries
    data = h.timeseries
    
    # Create identifiers for the 'uni' Universe
    uni_identifiers = [(res.segid, res.resid, res.resname, "residue") \
                       for res in uni.residues]
    
    # Create identifiers for the 'pdb' Universe (reference)
    identifiers = [(res.segid, res.resid, res.resname, "residue") \
                   for res in pdb.residues]
    
    # Map the identifiers of the uni Universe to their corresponding
    # indexes in the matrix
    uni_id2ix = \
        dict([(item, i) for i, item in enumerate(uni_identifiers)])
    
    # Utility function to get the identifier of a hydrogen bond
    get_identifier = \
        lambda uni, hbond: frozenset(((uni.atoms[hbond[0]].segid,
                                       uni.atoms[hbond[0]].resid,
                                       uni.atoms[hbond[0]].resname,
                                       "residue"),
                                      (uni.atoms[hbond[1]].segid,
                                       uni.atoms[hbond[1]].resid,
                                       uni.atoms[hbond[1]].resname,
                                       "residue")))
    
    # Initialize the full matrix to None
    fullmatrix = None
    
    # Create the full matrix if requested
    if do_fullmatrix:
        fullmatrix = np.zeros((len(identifiers),len(identifiers)))
    
    # Create empty list for output
    table_out = []
    
    if perresidue or do_fullmatrix:
        
        # Get the number of frames in the trajectory
        numframes = len(uni.trajectory)
        
        # Set the output string format
        outstr_fmt = "{:s}{:d}:{:s}{:d}\t\t{3.2f}\n"
        
        # Generate a list of sets
        setlist = \
            list(set(zip(*list(zip(*list(itertools.chain(*data))))[0:2])))
        
        # Get a list of the unique hydrogen bonds identified (between
        # pairs of residues)
        identifiers_setlist = \
            list(set([get_identifier(uni, hbond) for hbond in setlist]))
        
        # Initialize an empty counter (number of occurrences
        # for each hydrogen bond)
        outdata = collections.Counter(\
            {identifier : 0 for identifier in identifiers_setlist})
        
        # For each frame
        for frame in data:
            # Update the counter for the occurences of each
            # hydrogen bond in this frame
            outdata.update(\
                set([get_identifier(uni, hbond) for hbond in frame]))
        
        # For each hydrogen bond identified in the trajectory
        for identifier, hb_occur in outdata.items():
            # Get the persistence of the hydrogen bond
            hb_pers = (float(hb_occur)/float(numframes))*100
            # Convert the identifier from a frozenset to a list
            # to get items by index (the order does not matter)
            identifier_as_list = list(identifier)
            # Get info about the first residue
            res1 = identifier_as_list[0]
            res1_resix = uni_id2ix[res1]
            res1_segid, res1_resid = res1[:2]
            # Get info about the second residue (if present)
            res2 = \
                res1 if len(identifier) == 1 else identifier_as_list[1]
            res2_resix = uni_id2ix[res2]
            res2_segid, res2_resid = res2[:2]
            # Fill the full matrix if requested
            if do_fullmatrix:
                fullmatrix[res1_resix, res2_resix] = hb_pers
                fullmatrix[res2_resix, res1_resix] = hb_pers
            # Format the output string if requesets
            if perresidue:
                if hb_pers > perco:
                    outstr += outstr_fmt.format(res1_segid, res1_resid,
                                                res1_segid, res1_resid,
                                                hb_pers)
    
    # Do not merge hydrogen bonds per residue
    if not perresidue:
        
        # Utility function to get the identifier of a hydrogen bond
        get_list_identifier = \
            lambda uni, hbond: [(uni.atoms[hbond[0]].segid,
                                 uni.atoms[hbond[0]].resid,
                                 uni.atoms[hbond[0]].resname,
                                 "residue"),
                                (uni.atoms[hbond[1]].segid,
                                 uni.atoms[hbond[1]].resid,
                                 uni.atoms[hbond[1]].resname,
                                 "residue")]

        # Count hydrogen bonds by type
        table = h.count_by_type()
        hbonds_identifiers = \
            [get_list_identifier(uni, hbond) for hbond in table]
        
        # For each hydrogen bonds identified
        for i, hbidentifier in enumerate(hbonds_identifiers):
            
            # Get the hydrogen bond persistence
            hb_pers = table[i][-1]*100
            
            # Get donor heavy atom and acceptor atom
            donor_heavy_atom = (table[i][4],)
            acceptor_atom = (table[i][8],)
            
            # Consider only those hydrogen bonds whose persistence
            # is greater than the cut-off

            if hb_pers > perco:
                # Remove the res_tag columns
                res1 = identifiers[uni_id2ix[hbidentifier[0]]][0:3]
                res2 = identifiers[uni_id2ix[hbidentifier[1]]][0:3]
                persistence = (hb_pers,)
                row = res1 + donor_heavy_atom + res2 + \
                      acceptor_atom + persistence
                table_out.append(row)

    # Return table of hydrogen bonds and full matrix
    return table_out, fullmatrix
