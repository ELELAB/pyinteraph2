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

import sys
import logging as log
import collections
import itertools
import configparser as cp
import json
import struct
import numpy as np
import MDAnalysis as mda

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
        self.bins[''.join(bin[0:4])] = bin[4]
   
    def num_bins(self):
        return len(self.bins)

kbp_residues_list = ["ALA","ARG","ASN","ASP","CYS","GLN","GLU","GLY","HIS","ILE","LEU","LYS","MET","PHE","PRO","SER","THR","TRP","TYR","VAL"]

def parse_sparse(potential_file):
    header_fmt  = '400i'
    header_struct = struct.Struct(header_fmt)
    header_size = struct.calcsize(header_fmt)
    sparse_fmt  = '=iiiiiidddidxxxx'
    sparse_size = struct.calcsize(sparse_fmt)
    sparse_struct = struct.Struct(sparse_fmt)
    bin_fmt = '4cf'
    bin_size  = struct.calcsize(bin_fmt)
    bin_struct = struct.Struct(bin_fmt)
    pointer = 0
    sparses = []

    fh = open(potential_file, 'rb')
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
                tmp_struc = tuple([ s.decode('ascii') for s in tmp_struc[0:4] ] + [tmp_struc[4]])
                this_sparse.add_bin(tmp_struc)
                pointer += bin_size
            sparses[-1].append(this_sparse)
    
    if pointer != len(data):
        errstr = \
            "Error: could not completely parse the file {:s}" \
            " ({:d} bytes read, {:d} expected)"
        log.error(errstr.format(potential_file, pointer, len(data)))
        raise ValueError(errstr)

    sparses_dict = {}
    for i in range(len(kbp_residues_list)):
        sparses_dict[kbp_residues_list[i]] = {}
        for j in range(i):            
            sparses_dict[kbp_residues_list[i]][kbp_residues_list[j]] = {}
    for s in sparses:
        if s:
            sparses_dict[kbp_residues_list[s[0].r1]][kbp_residues_list[s[0].r2]] = s[0]
   
    logstr = "Done parsing file {:s}!"
    sys.stdout.write(logstr.format(potential_file))
    
    return sparses_dict


def parse_atomlist(fname):
    """Parse atom list for potential calculation."""

    try:
        fh = open(fname)
    except:
        raise IOError(f"Could not open file {kbpatomsfile}.")
    with fh:
        data = {}
        for line in fh:
            tmp = line.strip().split(":")
            data[tmp[0].strip()] = [i.strip() for i in tmp[1].split(",")]

    return data

def calc_potential(distances,
                   ordered_sparses,
                   kbT = 1.0):

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

def parse_cgs_file(fname):
    grps_str = 'CHARGED_GROUPS'
    res_str = 'RESIDUES'
    default_grps_str = 'default_charged_groups'

    cfg = cp.ConfigParser()
    cfg.read(fname)
    try:
        cfg.read(fname)
    except:
        log.error("file %s not readeable or not in the right format." % fname)
        exit(1)

    out = {}
    group_definitions = {}

    charged_groups = cfg.options(grps_str)
    charged_groups.remove(default_grps_str)
    charged_groups = [ i.strip() for i in charged_groups ]

    default_charged = cfg.get(grps_str, default_grps_str).split(",")
    default_charged = [ i.strip() for i in default_charged ]

    residues = cfg.options(res_str)

    for i in charged_groups + default_charged:
        group_definitions[i] = [s.strip() for s in cfg.get(grps_str, i).split(",")]
    for j in range(len(group_definitions[i])):
        group_definitions[i][j] = group_definitions[i][j].strip()

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
    except:
        logging.error("could not parse the charged groups file. Are there any inconsistencies?")

    return out

def parse_hbs_file(fname):
    hbs_str = 'HYDROGEN_BONDS'
    acceptors_str = 'ACCEPTORS'
    donors_str = 'DONORS'
    cfg = cp.ConfigParser()

    try:
        cfg.read(fname)
    except:
        log.error("file %s not readeable or not in the right format." % fname)

    acceptors = cfg.get(hbs_str, acceptors_str)
    tmp = acceptors.strip().split(",")
    acceptors = [ i.strip() for i in tmp ]

    donors = cfg.get(hbs_str, donors_str)
    tmp = donors.strip().split(",")
    donors = [ i.strip() for i in tmp ]

    return dict(ACCEPTORS=acceptors, DONORS=donors)

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

    log.info("Loading potential definition . . .")
    sparses = parse_sparse_func(potential_file)
    log.info("Loading input files...")

    ok_residues = []
    discarded_residues = set()
    residue_pairs = []
    atom_selections = []
    ordered_sparses = []
    numframes = len(uni.trajectory)

    for res in uni.residues:
        # check if the residue type is one of
        # those included in the list
        if res.resname in residues_list:
            # add it to the accepted residues
            ok_residues.append(res)
        else:
            # add it to the discarded residues
            discarded_residues.add(res)
            continue
        
        # for each other accepted residue
        for res2 in ok_residues[:-1]:
            res1 = res 
            seq_dist = abs(res1.ix - res2.ix)
            res1_segid = res1.segment.segid
            res2_segid = res2.segment.segid
            if not (seq_dist < seq_dist_co or res1_segid != res2_segid):
                # string comparison ?!
                if res2.resname < res1.resname:
                    res1, res2 = res2, res1
                
                this_sparse = sparses[res1.resname][res2.resname]
                
                # get the four atoms for the potential
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
                    # inform the user about the problem and
                    # continue
                    warnstr = \
                        "Could not identify essential atoms " \
                        "for the analysis ({:s}{:d}, {:s}{:d})"
                    log.warning(\
                        warnstr.format(res1.resname, \
                                       res1.resid, \
                                       res2.resname, \
                                       res2.resid))
                    
                    continue
                
                residue_pairs.append((res1, res2))
                atom_selections.append(selected_atoms)
                ordered_sparses.append(this_sparse)

    # create an matrix of floats to store scores (initially
    # filled with zeros)
    scores = np.zeros((len(residue_pairs)), dtype = np.float64)
    # set coordinates to None
    coords = None
    # for each frame in the trajectory
    numframe = 1
    for ts_i, ts in enumerate(uni.trajectory):
        # log the progress along the trajectory
        logstr = "Now analyzing: frame {:d} / {:d} ({:3.1f}%)\r"
        sys.stdout.write(\
            logstr.format(ts_i + 1, \
                          numframes, \
                          float(numframe)/float(numframes)*100.0))
        sys.stdout.flush()       
        
        # create an array of coordinates by concatenating the arrays of
        # atom positions in the selections row-wise
        coords = \
            np.array(\
                np.concatenate(\
                    [sel.positions for sel in atom_selections]), \
            dtype = np.float64)

        inner_loop = il.LoopDistances(coords, coords, None)
        # compute distances
        distances = \
            inner_loop.run_potential_distances(len(atom_selections), 4, 1)
        # compute scores
        scores += \
            calc_potential_func(distances = distances, \
                                ordered_sparses = ordered_sparses, \
                                kbT = kbT)
    
    # divide the scores for the lenght of the trajectory
    scores /= float(numframes)
    # create the output string
    outstr = ""
    # set the format for the representation of each pair of
    # residues in the output string
    outstr_fmt = "{:s}-{:s}{:d}:{:s}-{:s}{:d}\t{:.3f}\n"
    for i, score in enumerate(scores):
        if abs(score) > 0.000001:
            # update the output string
            outstr +=  \
                outstr_fmt.format(\
                    pdb.residues[residue_pairs[i][0].ix].segment.segid, \
                    pdb.residues[residue_pairs[i][0].ix].resname, \
                    pdb.residues[residue_pairs[i][0].ix].resid, \
                    pdb.residues[residue_pairs[i][1].ix].segment.segid, \
                    pdb.residues[residue_pairs[i][1].ix].resname, \
                    pdb.residues[residue_pairs[i][1].ix].resid, \
                    score)
    
    # inizialize the matrix to None  
    dm = None   
    if do_fullmatrix:
        # if requested, create the matrix
        dm = np.zeros((len(pdb.residues), len(pdb.residues)))
        # use numpy "fancy indexing" to fill the matrix
        # with scores at the corresponding residues pairs
        # positions
        pair_firstelems = [pair[0].ix for pair in residue_pairs]
        pairs_secondelems = [pair[1].ix for pair in residue_pairs]
        dm[pair_firstelems, pairs_secondelems] = scores
        dm[pairs_secondelems, pair_firstelems] = scores
    
    # return the output string and the matrix
    return (outstr, dm)

def calc_dist_matrix(uni, \
                     idxs, \
                     chosenselections, \
                     co, \
                     mindist = False, \
                     mindist_mode = None, \
                     pos_char = "p", \
                     neg_char = "n"):
    """Compute matrix of distances"""
    
    numframes = len(uni.trajectory)
    # initialize the final matrix
    percmat = \
        np.zeros((len(chosenselections), len(chosenselections)), \
                 dtype = np.float64)

    if mindist:
        # lists for positively charged atoms
        pos = []
        pos_idxs = []
        pos_sizes = []
        # lists for negatively charged atoms
        neg = []
        neg_idxs = []
        neg_sizes = []

        for i in range(len(idxs)):
            # character in the index indicating the
            # charge of the atom
            charge_char = idxs[i][3][-1]
            # if the index contains the indication of a
            # positively charged atom
            if charge_char == pos_char:
                pos.append(chosenselections[i])
                pos_idxs.append(idxs[i])
                pos_sizes.append(len(chosenselections[i]))
            # if the index contains the indication of a
            # negatively charged atom
            elif charge_char == neg_char:
                neg.append(chosenselections[i])
                neg_idxs.append(idxs[i])
                neg_sizes.append(len(chosenselections[i]))
            # if none of the above
            else: 
                errstr = \
                    "Accepted values are either '{:s}' or '{:s}', " \
                    "but {:s} was found."
                raise ValueError(errstr.format(pos_char, \
                                               neg_char, \
                                               charge_char))

        # convert lists of positions into arrays
        pos_sizes = np.array(pos_sizes, dtype = np.int)
        neg_sizes = np.array(neg_sizes, dtype = np.int)

        # if we are interested in interactions between atoms
        # with different charges
        if mindist_mode == "diff":
            sets = [(pos, neg)]
            sets_idxs = [(pos_idxs, neg_idxs)]
            sets_sizes = [(pos_sizes, neg_sizes)]
        # if we are interested in interactions between atoms
        # with the same charge
        elif mindist_mode == "same":
            sets = [(pos, pos), (neg, neg)]
            sets_idxs = [(pos_idxs, pos_idxs), (neg_idxs, neg_idxs)]
            sets_sizes = [(pos_sizes, pos_sizes), (neg_idxs, neg_sizes)]
        # if we are interested in both
        elif mindist_mode == "both":
            sets = [(chosenselections, chosenselections)]
            sets_idxs = [(idxs, idxs)]
            sizes =  [len(s) for s in chosenselections]
            sets_sizes = [(sizes, sizes)]
        # unrecognized choice             
        else:
            choices = ["diff", "same", "both"]
            errstr = \
                "Accepted values for 'mindist_mode' are {:s}, " \
                "but {:s} was found."
            raise ValueError(errstr.format(", ".join(choices), \
                                           mindist_mode))

        percmats = []
        # create an empty list to store the atomic coordinates
        coords = [([[], []]) for s in sets]

        # for each frame in the trajectory
        numframe = 1
        for ts in uni.trajectory:
            # log the progress along the trajectory
            logstr = \
                "Caching coordinates: frame {:d} / {:d} ({:3.1f}%)\r"
            sys.stdout.write(logstr.format(\
                                numframe, \
                                numframes, \
                                float(numframe)/float(numframes)*100.0))
            sys.stdout.flush()
            # update the frame number
            numframe += 1
            
            # for each set of atoms
            for s_index, s in enumerate(sets):
                if s[0] == s[1]:
                    # triangular case
                    log.info("Caching coordinates...")
                    for group in s[0]:
                        coords[s_index][0].append(group.positions)
                        coords[s_index][1].append(group.positions)
                else:
                    # square case
                    log.info("Caching coordinates...")
                    for group in s[0]:
                        coords[s_index][0].append(group.positions)
                    for group in s[1]:
                        coords[s_index][1].append(group.positions)

        for s_index, s in enumerate(sets):
            # recover the final matrix
            if s[0] == s[1]:
                # triangular case
                this_coords = \
                    np.array(np.concatenate(coords[s_index][0]), \
                             dtype = np.float64)
                # compute the distances within the cut-off
                inner_loop = il.LoopDistances(this_coords, this_coords, co)
                percmats.append(\
                    inner_loop.run_triangular_mindist(\
                        sets_sizes[s_index][0]))

            else:
                # square case
                this_coords1 = \
                    np.array(np.concatenate(coords[s_index][0]), \
                             dtype = np.float64)              
                this_coords2 = \
                    np.array(np.concatenate(coords[s_index][1]), \
                             dtype = np.float64)
                # compute the distances within the cut-off
                inner_loop = il.LoopDistances(this_coords1, this_coords2, co)
                percmats.append(\
                    inner_loop.run_square_mindist(\
                        sets_sizes[s_index][0], \
                        sets_sizes[s_index][1]))

        for s_index, s in enumerate(sets): 
            # recover the final matrix
            pos_idxs = sets_idxs[s_index][0]
            neg_idxs = sets_idxs[s_index][1]
            if s[0] == s[1]:
                # triangular case
                for j in range(len(s[0])):
                    for k in range(0, j):
                        ix_j = idxs.index(pos_idxs[j])
                        ix_k = idxs.index(pos_idxs[k])
                        percmat[ix_j, ix_k] = percmats[s_index][j,k]         
                        percmat[ix_k, ix_j] = percmats[s_index][j,k]
            else: 
                # square case
                for j in range(len(s[0])):
                    for k in range(len(s[1])):
                        ix_j_p = idxs.index(pos_idxs[j])
                        ix_k_n = idxs.index(neg_idxs[k])
                        percmat[ix_j_p, ix_k_n] = percmats[s_index][j,k]         
                        percmat[ix_k_n, ix_j_p] = percmats[s_index][j,k]
                     
    else:
        # empty list of matrices of centers of mass
        all_coms = []
        # for each frame in the trajectory
        numframe = 1
        for ts in uni.trajectory:
            # log the progress along the trajectory
            logstr = "Now analyzing: frame {:d} / {:d} ({:3.1f}%)\r"
            sys.stdout.write(logstr.format(\
                                numframe, \
                                numframes, \
                                float(numframe)/float(numframes)*100.0))
            sys.stdout.flush()
            # update the frame number
            numframe += 1
            
            # matrix of centers of mass for the chosen selections
            coms_list = [sel.center(sel.masses) for sel in chosenselections]
            coms = np.array(coms_list, dtype = np.float64)
            all_coms.append(coms)

        # create a matrix of all centers of mass along the trajectory
        all_coms = np.concatenate(all_coms)
        # compute the distances within the cut-off
        inner_loop = il.LoopDistances(all_coms, all_coms, co)
        percmat = inner_loop.run_triangular_distmatrix(coms.shape[0])
    
    # convert the matrix into an array
    percmat = np.array(percmat, dtype = np.float64)/numframes*100.0

    return percmat


def assign_ff_masses(ffmasses, chosenselections):
    # load force field data
    with open(ffmasses, 'r') as fh:
        ffdata = json.load(fh)
    for selection in chosenselections:
        # for each atom
        for atom in selection:
            atom_resname = atom.residue.resname
            atom_resid = atom.residue.resid
            atom_name = atom.name
            try:                
                atom.mass = ffdata[1][atom_resname][atom_name]
            except:
                warnstr = \
                    "Atom type not recognized (resid {:d}, " \
                    "resname {:s}, atom {:s}). " \
                    "Atomic mass will be guessed."
                log.warning(warnstr.format(atom_resid, \
                                           atom_resname, \
                                           atom_name))   

def generate_cg_identifiers(pdb, uni, **kwargs):
    """Generate charged atoms identifiers."""

    cgs = kwargs["cgs"]
    # preprocess CGs: divide wolves and lambs
    filter_func_p = lambda x: not x.startswith("!")
    filter_func_n = lambda x:     x.startswith("!")   

    for res, dic in cgs.items(): 
        for cgname, cg in dic.items():
            # True : atoms that must exist (negative of negative)
            true_set = set(filter(filter_func_p, cg))
            # False: atoms that must NOT exist (positive of negative)
            false_set = set([j[1:] for j in filter(filter_func_n, cg)])
            # update CGs
            cgs[res][cgname] =  {True : true_set, False : false_set}
    
    # list of identifiers
    identifiers = [(r.segid, r.resid, r.resname, "") for r in pdb.residues]
    # empty lists of IDs and atom selections
    idxs = []
    chosenselections = []
    # atom selection string
    selstring = "segid {:s} and resid {:d} and (name {:s})"
    # for each residue in the Universe
    for k, res in enumerate(uni.residues):
        segid = res.segid
        resname = res.resname
        resid = res.resid
        try:
            cgs_items = cgs[resname].items()
        except KeyError:
            logstr = \
                "Residue {:s} is not in the charge recognition set. " \
                "Will be skipped."
            log.warn(logstr.format(resname))
            continue
        # current atom names
        setcurnames = set(res.atoms.names)
        for cgname, cg in cgs_items:
            # get the set of atoms to be kept
            atoms_must_exist = cg[True]
            atoms_must_not_exist = cg[False]
            # set the condition to keep atoms in the current
            # residue, i.e. the atoms that must be present are
            # present and those which must not be present are not

            condition_to_keep = \
                atoms_must_exist.issubset(setcurnames) and \
                atoms_must_not_exist.isdisjoint(setcurnames)
            # if the condition is met
            if condition_to_keep:
                idx = (pdb.residues[k].segment.segid,
                       pdb.residues[k].resid,
                       pdb.residues[k].resname,
                       cgname)
                idxs.append(idx)

                atoms_str = " or name ".join(atoms_must_exist)
                selection = \
                    uni.select_atoms(selstring.format(\
                        segid, resid, atoms_str))                
                # update lists of IDs and atom selections
                chosenselections.append(selection)
                # log the selection
                atoms_names_str = ", ".join([a.name for a in selection])
                logstr = "{:d} {:s} ({:s})"
                log.info(logstr.format(resid, resname, atoms_names_str))
            else:
                pass

    # return identifiers, IDs and atom selections
    return (identifiers, idxs, chosenselections)
      
def generate_sc_identifiers(pdb, uni, **kwargs):
    """Generate side chain identifiers"""

    # get the residue names list
    reslist = kwargs["reslist"]
    # log the list of residue names
    log.info("Selecting residues: {:s}".format(", ".join(reslist)))
    # backbone atoms must be excluded
    backbone_atoms = \
        ["CA", "C", "O", "N", "H", "H1", "H2", \
         "H3", "O1", "O2", "OXT", "OT1", "OT2"]
    # create list of identifiers
    identifiers = \
        [(r.segid, r.resid, r.resname, "sidechain") for r in pdb.residues]
    # create empty lists for IDs and atom selections
    chosenselections = []
    idxs = []
    # atom selection string
    selstring = "segid {:s} and resid {:d} and (name {:s})"
    # start logging the chosen selections
    log.info("Chosen selections:")
    # for each residue name in the residue list
    for resname_in_list in reslist:
        resname_3letters = resname_in_list[0:3]
        # update the list of IDs with all those residues matching
        # the current residue type
        for identifier in identifiers:
            if identifier[2] == resname_3letters:
                idxs.append(identifier)
        # for each residue in the Universe
        for res in uni.residues:
            if res.resname[0:3] == resname_in_list:
                resid = res.resid
                segid = res.segid
                resname = res.resname
                # get side chain atom names
                sc_atoms_names = \
                    [a.name for a in res.atoms \
                     if a.name not in backbone_atoms]
                sc_atoms_str = " or name ".join(sc_atoms_names)
                sc_atoms_names_str = ", ".join(sc_atoms_names)
                # get the side chain atom selection
                selection = \
                    uni.select_atoms(selstring.format(\
                        segid, resid, sc_atoms_str))
                chosenselections.append(selection)
                # log the selection
                log.info("{:d} {:s} ({:s})".format(\
                    resid, resname, sc_atoms_names_str))

    return (identifiers, idxs, chosenselections)
     
def calc_sc_fullmatrix(identifiers, idxs, percmat, perco):
    """Calculate side chain-side chain interaction matrix
    (hydrophobic contacts)"""
    # create a matrix of size identifiers x identifiers
    fullmatrix = np.zeros((len(identifiers), len(identifiers)))
    # get where (index) the elements of idxs are in identifiers
    
    where_idxs_in_identifiers = \
        [identifiers.index(item) for item in idxs]
    # get where (index) each element of idxs is in idxs
    where_idxs_in_idxs = [i for i, item in enumerate(idxs)]
    # get where (i,j coordinates) each element of idxs is in
    # fullmatrix
    positions_identifiers_in_fullmatrix = \
        itertools.combinations(where_idxs_in_identifiers, 2)
    # get where (i,j coordinates) each element of idxs is in
    # percmat (which has dimensions len(idxs) x len(idxs))
    positions_idxs_in_percmat = \
        itertools.combinations(where_idxs_in_idxs, 2)
    # unpack all pairs of i,j coordinates in lists of i 
    # indexes and j indexes
    i_fullmatrix, j_fullmatrix = zip(*positions_identifiers_in_fullmatrix)
    i_percmat, j_percmat = zip(*positions_idxs_in_percmat)
    # use numpy "fancy indexing" to fill fullmatrix with the
    # values in percmat corresponding to each pair of elements
    fullmatrix[i_fullmatrix, j_fullmatrix] = percmat[i_percmat,j_percmat]
    fullmatrix[j_fullmatrix, i_fullmatrix] = percmat[i_percmat,j_percmat]
    # return the full matrix (square matrix)
    return fullmatrix

def calc_cg_fullmatrix(identifiers, idxs, percmat, perco):
    """Calculate charged atoms interaction matrix (salt bridges)"""
    
    # search for residues with duplicate ID
    duplicates = []
    idx_index = 0
    while idx_index < percmat.shape[0]:
        # function to retrieve only residues whose ID (segment ID,
        # residue ID and residue name) perfectly matches the one of
        # the residue currently under evaluation (ID duplication)
        filter_func = lambda x: x[0:3] == idxs[idx_index][0:3]
        # get where (indexes) residues with the same ID as that
        # currently evaluated are
        rescgs = list(map(idxs.index, filter(filter_func, idxs)))
        # save the indexes of the duplicate residues
        duplicates.append(frozenset(rescgs))
        # update the counter
        idx_index += len(rescgs)
    
    # if no duplicates are found, the corrected matrix will have
    # the same size as the original matrix 
    corrected_percmat = np.zeros((len(duplicates), len(duplicates)))
    # generate all the possible combinations of the sets of 
    # duplicates (each set represents a residue who found
    # multiple times in percmat)
    dup_combinations = itertools.combinations(duplicates, 2)
    # for each pair of sets of duplicates
    for dup_resi, dup_resj in dup_combinations:
        # get where residue i and residue j should be uniquely
        # represented in the corrected matrix
        corrected_ix_i = duplicates.index(dup_resi)
        corrected_ix_j = duplicates.index(dup_resj)
        # get the positions of all interactions made by each instance
        # of residue i with each instance of residue j in percmat
        index_i, index_j = zip(*itertools.product(dup_resi,dup_resj))
        # use numpy "fancy indexing" to put in corrected_percmat
        # only the strongest interaction found between instances of
        # residue i and instances of residue j
        corrected_percmat[corrected_ix_i,corrected_ix_j] = \
            np.max(percmat[index_i,index_j])
    
    # to generate the new IDs, get the first instance of each residue
    # (duplicates all share the same ID) and add an empty string as
    # last element of the ID
    corrected_idxs = \
        [idxs[list(i)[0]][0:3] + ("",) for i in list(duplicates)]
    # create a matrix of size identifiers x identifiers
    fullmatrix = np.zeros((len(identifiers), len(identifiers)))
    # get where (index) the elements of corrected_idxs are in identifiers
    where_idxs_in_identifiers = \
        [identifiers.index(item) for item in corrected_idxs]
    # get where (index) each element of corrected_idxs 
    # is in corrected_idxs
    where_idxs_in_idxs = [i for i, item in enumerate(corrected_idxs)]
    # get where (i,j coordinates) each element of corrected_idxs
    # is in fullmatrix
    positions_identifiers_in_fullmatrix = \
        itertools.combinations(where_idxs_in_identifiers, 2)
    # get where (i,j coordinates) each element of corrected_idxs
    # is in corrected_percmat
    positions_idxs_in_corrected_percmat = \
        itertools.combinations(where_idxs_in_idxs, 2)
    # unpack all pairs of i,j coordinates in lists of i 
    # indexes and j indexes
    i_fullmatrix, j_fullmatrix = \
        zip(*positions_identifiers_in_fullmatrix)
    i_corrected_percmat, j_corrected_percmat = \
        zip(*positions_idxs_in_corrected_percmat)
    # use numpy "fancy indexing" to fill fullmatrix with the
    # values in percmat corresponding to each pair of elements
    fullmatrix[i_fullmatrix, j_fullmatrix] = \
        corrected_percmat[i_corrected_percmat, j_corrected_percmat]
    fullmatrix[j_fullmatrix, i_fullmatrix] = \
        corrected_percmat[i_corrected_percmat, j_corrected_percmat]
    # return the full matrix (square matrix)
    return fullmatrix


def create_table_list(contact_list, hb=False):
    """Takes in a list of tuples and returns a list of arrays where the first
    array contains all contacts and subsequent arrays are split by chain"""
    array = np.array(contact_list)
    output = []
    output.append(array)
    array_T = array.T
    chains = np.unique(array_T[0])
    #choose which column has the second residue
    if hb:
        sec_res = 4
    else:
        sec_res = 3
    #if multiple chains are present, split the contacts by chain
    if len(chains) > 1:
	#for each chain, add the nodes that are only in contact with the same chain
        for i in range(len(chains)):
            logical_vector = np.logical_and(chains[i] == array_T[0], chains[i] == array_T[sec_res])
            output_chain = array[logical_vector]
            output.append(output_chain)
        #add all nodes that are in contact with different chains
        logical_vector = array_T[0] != array_T[sec_res]
        output_diff_chain = array[logical_vector]
        output.append(output_diff_chain)
    #remove any empty arrays
    output = [array for array in output if array.shape[0] != 0]
    return(output)

def create_matrix_list(full_matrix, table_list, pdb, hb = False):
    """Takes in the full matrix and returns a list of matrices split by chain.
    Returns list of size 1 if only one chain."""
    mat_list = []
    mat_list.append(full_matrix)
    if hb:
        sec_res_id = 5
    else:
        sec_res_id = 4
    if len(table_list) > 1:
        mat_len = full_matrix.shape[0]
        #map each residue id to a position in the matrix
        resids = pdb.residues.resids
        res_dict = {str(resids[i]):i for i in range(mat_len)}
        #for all tables excluding the first one, create an empty
        #matrix and fill it with the appropriate values
        for i in range(1, len(table_list)):
            matrix = np.zeros((mat_len, mat_len))
            for row in table_list[i]:
                mat_i = res_dict[row[1]]
                mat_j = res_dict[row[sec_res_id]]
                matrix[mat_i][mat_j] = full_matrix[mat_i][mat_j]
                matrix[mat_j][mat_i] = full_matrix[mat_j][mat_i]
            mat_list.append(matrix)
    return(mat_list)


def save_output_list(table_list, filename, mat_list=None, hb=False):
    """Save each item in the list as a separate file. The first
    item in the list contains the whole dataset, the last item contains
    interchain data. Rest are intrachain"""
    if mat_list is not None:
        ext = ".mat"
        data = mat_list
        delim = ''
        format = "%.1f"
    else:
        ext = ".csv"
        data = table_list
        delim = ','
        format = '%s'
    for i in range(len(table_list)):
        if i == 0:
            np.savetxt(filename + "_all_bonds" + ext, data[i], delimiter=delim, fmt=format)
        else:
            chain1 = table_list[i][0][0]
            if hb:
                sec_res = 4
            else:
                sec_res = 3
            chain2 = table_list[i][0][sec_res]
            if chain1 == chain2:
                np.savetxt(filename + "_intra_chain_" + str(chain1) + ext, data[i], delimiter=delim, fmt=format)
            else:
                np.savetxt(filename + "_inter_chain"+ ext, data[i], delimiter=delim, fmt=format)



############################ INTERACTIONS #############################

def do_interact(identfunc, \
                pdb, \
                uni, \
                co = 5.0, \
                perco = 0.0, \
                assignffmassesfunc = assign_ff_masses, \
                distmatrixfunc = calc_dist_matrix, \
                ffmasses = None, \
                fullmatrixfunc = None, \
                mindist = False, \
                mindist_mode = None, \
                **identargs):
    
    # get identifiers, indexes and atom selections
    identifiers, idxs, chosenselections = identfunc(pdb, uni, **identargs)

    # assign atomic masses to atomic selections if not provided
    if ffmasses is None:
        log.info("No force field assigned: masses will be guessed.")
    else:
        try:
            assignffmassesfunc(ffmasses, chosenselections)
        except IOError:
            logstr = "Force field file not found or not readable. " \
                     "Masses will be guessed."
            log.warning(logstr)     
    # calculate the matrix of persistences
    percmat = calc_dist_matrix(uni = uni, \
                               idxs = idxs,\
                               chosenselections = chosenselections, \
                               co = co, \
                               mindist = mindist, \
                               mindist_mode = mindist_mode)
    # get shortened indexes and identifiers
    short_idxs = [i[0:3] for i in idxs]
    short_ids = [i[0:3] for i in identifiers]
    # create empty list for output
    contact_list = []
    # get where in the lower triangle of the matrix (it is symmeric)
    # the value is greater than the persistence cut-off
    where_gt_perco = np.argwhere(np.tril(percmat>perco))
    for i, j in where_gt_perco:
        #print(i, j)
        #print("IDXSi", short_idxs[i])
        #print("IDXSj", short_idxs[j])

        res1 = short_ids[short_ids.index(short_idxs[i])]
        res2 = short_ids[short_ids.index(short_idxs[j])]
        persistence = (percmat[i,j],)
        contact_list.append(res1 + res2 + persistence)
    # set the full matrix to None
    fullmatrix = None
    # compute the full matrix if requestes
    if fullmatrixfunc is not None:
        fullmatrix = fullmatrixfunc(identifiers = identifiers, \
                                    idxs = idxs, \
                                    percmat = percmat, \
                                    perco = perco)
    
    # return output list and fullmatrix
    return (contact_list, fullmatrix)

############################### HBONDS ################################

def do_hbonds(sel1, \
              sel2, \
              pdb, \
              uni, \
              update_selection1 = True, \
              update_selection2 = True, \
              filter_first = False, \
              distance = 3.0, \
              angle = 120, \
              perco = 0.0, \
              perresidue = False, \
              do_fullmatrix = False, \
              other_hbs = None):
    
    # import the hydrogen bonds analysis module
    from MDAnalysis.analysis.hbonds import hbond_analysis
    # check if selection 1 is valid
    try:
        sel1atoms = uni.select_atoms(sel1)
    except:
        log.error("ERROR: selection 1 is invalid")
    # check if selection 2 is valid
    try:
        sel2atoms = uni.select_atoms(sel2)
    except:
        log.error("ERROR: selection 2 is invalid")      
    # check if custom donors and acceptors were provided
    if other_hbs is None:
        class Custom_HydrogenBondAnalysis(hbond_analysis.HydrogenBondAnalysis):
            pass
        hb_ff = "CHARMM27"
    else:  
        # custom names
        class Custom_HydrogenBondAnalysis(hbond_analysis.HydrogenBondAnalysis):
            DEFAULT_DONORS = {"customFF" : other_hbs["DONORS"]}
            DEFAULT_ACCEPTORS = {"customFF" : other_hbs["ACCEPTORS"]}
        hb_ff = "customFF"
    # set up the hydrogen bonds analysis
    h = Custom_HydrogenBondAnalysis(universe = uni, \
                                    selection1 = sel1, \
                                    selection2 = sel2, \
                                    distance = distance, \
                                    angle = angle, \
                                    forcefield = hb_ff, \
                                    update_selection1 = update_selection1, \
                                    update_selection2 = update_selection2, \
                                    filter_first = filter_first)
    # inform the user about the hydrogen bond analysis parameters
    logstr = "Will use {:s}: {:s}"
    log.info(logstr.format("acceptors", \
                           ", ".join(h.DEFAULT_ACCEPTORS[hb_ff])))
    log.info(logstr.format("donors", \
                           ", ".join(h.DEFAULT_DONORS[hb_ff])))
    log.info("Running hydrogen bonds analysis . . .")
    # run the hydrogen bonds analysis
    h.run()
    log.info("Done! Finalizing . . .")
    # get the hydrogen bonds timeseries
    data = h.timeseries
    # create identifiers for the uni Universe
    uni_identifiers = [(res.segid, res.resid, res.resname, "residue") \
                       for res in uni.residues]
    # create identifiers for the pdb Universe (reference)
    identifiers = [(res.segid, res.resid, res.resname, "residue") \
                   for res in pdb.residues]
    # map the identifiers of the uni Universe to their corresponding
    # indexes in the matrix
    uni_id2ix = \
        dict([(item, i) for i, item in enumerate(uni_identifiers)])
    # utility function to get the identifier of a hydrogen bond
    get_identifier = \
        lambda uni, hbond: frozenset(((uni.atoms[hbond[0]].segid, \
                                       uni.atoms[hbond[0]].resid, \
                                       uni.atoms[hbond[0]].resname, \
                                       "residue"), \
                                      (uni.atoms[hbond[1]].segid, \
                                       uni.atoms[hbond[1]].resid, \
                                       uni.atoms[hbond[1]].resname, \
                                       "residue")))
    # initialize the full matrix to None
    fullmatrix = None
    # create the full matrix if requested
    if do_fullmatrix:
        fullmatrix = np.zeros((len(identifiers),len(identifiers)))
    # create empty list for output
    contact_list = []
    if perresidue or do_fullmatrix:
        # get the number of frames in the trajectory
        numframes = len(uni.trajectory)
        # compatible with Python 3
        setlist = \
            list(set(zip(*list(zip(*list(itertools.chain(*data))))[0:2])))
        # get a list of the unique hydrogen bonds identified (between
        # pairs of residues)
        identifiers_setlist = \
            list(set([get_identifier(uni, hbond) for hbond in setlist]))
        # initialize an empty counter (number of occurrences
        # for each hydrogen bond)
        outdata = collections.Counter(\
            {identifier : 0 for identifier in identifiers_setlist})
        # for each frame
        for frame in data:
            # update the counter for the occurences of each
            # hydrogen bond in this frame
            outdata.update(\
                set([get_identifier(uni, hbond) for hbond in frame]))
        # for each hydrogen bond identified in the trajectory
        for identifier, hb_occur in outdata.items():
            # get the persistence of the hydrogen bond
            hb_pers = (float(hb_occur)/float(numframes))*100
            # convert the identifier from a frozenset to a list
            # to get items by index (the order does not matter)
            identifier_as_list = list(identifier)
            # get info about the first residue
            res1 = identifier_as_list[0]
            res1_resix = uni_id2ix[res1]

            res1_segid, res1_resid = res1[:2]
            # get info about the second residue (if present)
            res2 = res1 if len(identifier) == 1 else identifier_as_list[1]
            res2_resix = uni_id2ix[res2]
            res2_segid, res2_resid = res2[:2]
            # fill the full matrix if requested
            if do_fullmatrix:
                fullmatrix[res1_resix, res2_resix] = hb_pers
                fullmatrix[res2_resix, res1_resix] = hb_pers
            # format the output string if requesets
            if perresidue:
                if hb_pers > perco:
                    outstr += outstr_fmt.format(res1_segid, res1_resid, \
                                                res1_segid, res1_resid, \
                                                hb_pers)
    
    # do not merge hydrogen bonds per residue
    if not perresidue:
            # utility function to get the identifier of a hydrogen bond
        get_list_identifier = \
            lambda uni, hbond: [ (uni.atoms[hbond[0]].segid, \
                                           uni.atoms[hbond[0]].resid, \
                                           uni.atoms[hbond[0]].resname, \
                                           "residue"), \
                                          (uni.atoms[hbond[1]].segid, \
                                           uni.atoms[hbond[1]].resid, \
                                           uni.atoms[hbond[1]].resname, \
                                           "residue") ]

        # count hydrogen bonds by type
        table = h.count_by_type()

        hbonds_identifiers = \
            [ get_list_identifier(uni, hbond) for hbond in table ]
        # set output string format
        outstr_fmt = \
            "{:s}-{:d}{:s}_{:s}:{:s}-{:d}{:s}_{:s}\t\t{:3.2f}\n"
        # for each hydrogen bonds identified
        for i, hbidentifier in enumerate(hbonds_identifiers):
            # get the hydrogen bond persistence
            hb_pers = table[i][-1]*100
            # get donor heavy atom and acceptor atom
            donor_heavy_atom = table[i][4]
            acceptor_atom = table[i][8]
            # consider only those hydrogen bonds whose persistence
            # is greater than the cut-off

            if hb_pers > perco:
                res1 = identifiers[uni_id2ix[hbidentifier[0]]]
                res2 = identifiers[uni_id2ix[hbidentifier[1]]]
                persistence = (hb_pers,)
                contact_list.append(res1 + res2 + persistence)

    # return contact list  and full matrix
    return (contact_list, fullmatrix)
