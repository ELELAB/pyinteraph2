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

from __future__ import absolute_import
import sys
import logging as log

vinfo = sys.version_info
if vinfo[0] < 2 or (vinfo[0] == 2 and vinfo[1] < 7):
    errstr = \
        "Your Python version is {:s}, but only " \
        "versions >= 2.7 are supported."
    log.error(errstr.format(sys.version))
    exit(1)

import json
import struct

import numpy as np
import MDAnalysis as mda
from MDAnalysis.analysis.distances import distance_array

from itertools import chain
from innerloops import LoopDistances


class LoopBreakError(Exception):
    pass

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

    def add_bins(self, bins):
        for i in bins:
            self.bins[str(bins[0:4])] = bins[4]
    
    def num_bins(self):
        return len(self.bins)


def parse_sparse(potential_file, residues_list):
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

    fh = open(potential_file)
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
                this_sparse.add_bin(\
                    bin_struct.unpack(\
                        data[pointer : pointer + bin_size]))
                pointer += bin_size
            sparses[-1].append(this_sparse)
    
    if pointer != len(data):
        errstr = \
            "Error: could not completely parse the file {:s}" \
            " ({:d} bytes read, {:d} expected)"
        log.error(errstr.format(potential_file, pointer, len(data)))
        exit(1)

    sparses_dict = {}
    for i in range(len(residues_list)):
        sparses_dict[residues_list[i]] = {}
        for j in range(i):            
            sparses_dict[residues_list[i]][residues_list[j]] = {}
            
    for s in sparses:
        if s:
            sparses_dict[residues_list[s[0].r1]][residues_list[s[0].r2]] = s[0]
    
    logstr = "Done parsing file {:s}!"
    sys.stodout.write(logstr.format(potential_file))
    
    return sparses_dict


def parse_atomlist(fname):
    try:
        fh = open(fname)
    except:
        errstr = "Could not open file {:s}. Exiting..."
        log.error(errstr.format(fname), exc_info = True)
        exit(1)
    
    data = {}
    for line in fh:
        tmp = line.strip().split(":")
        data[tmp[0].strip()] = [i.strip() for i in tmp[1].split(",")]
    
    fh.close()
    return data    


def do_potential(kbp_atomlist, \
                residues_list, \
                potential_file, \
                seq_dist_co = 0, \
                grof = None, \
                xtcf = None, \
                pdbf = None, \
                uni = None, \
                pdb = None, \
                dofullmatrix = True, \
                kbT = 1.0):

    log.info("Loading potential definition . . .")
    sparses = parse_sparse(potential_file, residues_list)
    log.info("Loading input files...")

    if pdb is None or uni is None:
        if pdbf is None or grof is None or xtcf is None:
            errstr =\
                "If you do not pass 'pdb' or 'uni', you have to " \
                "pass 'pdbf', 'grof' and 'xtcf'"
            raise ValueError(errstr)
        
        pdb, uni = load_sys(pdbf, grof, xtcf)

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
            ok_residues.append(res)
        else:
            discarded_residues.add(res)
            continue
        
        for secondres in ok_residues[:-1]:
            firstres = res 
            
            if not (abs(res.ix - secondres.ix) < seq_dist_co \
            or firstres.segment.segid != secondres.segment.segid):
                # string comparison ?!
                if secondres.resname < firstres.resname:
                    ii,j = j,ii
                
                this_sparse = sparses[firstres.resname][secondres.resname]
                
                atom0, atom1, atom2, atom3 = \
                    (kbp_atomlist[firstres.resname][this_sparse.p1_1],
                     kbp_atomlist[firstres.resname][this_sparse.p1_2],
                     kbp_atomlist[secondres.resname][this_sparse.p2_1],
                     kbp_atomlist[secondres.resname][this_sparse.p2_2])
                
                try:
                    index_atom0 = \
                        firstres.atoms.names.tolist().index(atom0)
                    index_atom1 = \
                        firstres.atoms.names.tolist().index(atom1)
                    index_atom2 = \
                        secondres.atoms.names.tolist().index(atom2)
                    index_atom3 = \
                        secondres.atoms.named.tolist().index(atom3)
                    selected_atoms = \
                        mda.core.groups.AtomGroup(\
                             firstres.atoms[index_atom0],
                             firstres.atoms[index_atom1],
                             secondres.atoms[index_atom2],
                             secondres.atoms[index_atom3])
                except:
                    warnstr = \
                        "Could not identify essential atoms " \
                        "for the analysis ({:s}{:d}, {:s}{:d})"
                    log.warning(\
                        warnstr.format(firstres.resname, \
                                       firstres.resid, \
                                       secondres.resname, \
                                       secondres.resid))
                    
                    continue
                
                residue_pairs.append((firstres, secondres))
                atom_selections.append(selected_atoms)
                ordered_sparses.append(this_sparse)

    scores = np.zeros((len(residue_pairs)), dtype = float)  
    coords = None
    
    for ts in uni.trajectory:
        logstr = "Now analyzing: frame {:d} / {:d} ({:3.1f}%)\r"
        sys.stdout.write(logstr.format(a, \
                                       numframes, \
                                       float(a)/float(numframes)*100.0))
        sys.stdout.flush()       
   
        coords = np.array(\
            np.concatenate(\
                [sel.positions for sel in atom_selections]), \
            dtype = np.float64)

        inner_loop = LoopDistances(coords, coords, None)
        distances = \
            inner_loop.run_potential_distances(\
                len(atom_selections), 4, 1)

        scores += \
            calc_potential(distances = distances, \
                           ordered_sparses = ordered_sparses, \
                           pdb = pdb, \
                           uni = uni, \
                           distco = seq_dist_co, \
                           kbT = kbT)
    
    # divide the scores for the lenght of the trajectory
    scores /= float(len(uni.trajectory))
    outstr = ""
    outstr_fmt = "{:s}-{:s}{:s}:{:s}-{:s}{:s}\t{:.3f}\n"
    for i, score in enumerate(scores):
        if abs(score) > 0.000001:
            outstr +=  \
                outstr_fmt.format(\
                    pdb.residues[residue_pairs[i][0]].segment.segname, \
                    pdb.residues[residue_pairs[i][0]].resname, \
                    pdb.residues[residue_pairs[i][0]].resid, \
                    pdb.residues[residue_pairs[i][1]].segment.segname, \
                    pdb.residues[residue_pairs[i][1]].resname, \
                    pdb.residues[residue_pairs[i][1]].resid, \
                    score)
    
    # inizialize the matrix to None  
    dm = None   
    if dofullmatrix:
        # if requested, create the matrix
        dm = np.zeros((len(pdb.residues), len(pdb.residues)))
        # use numpy "fancy indexing" to fill the matrix
        # with scores at the corresponding residues pairs
        # positions
        pair_firstelems = [pair[0] for pair in residue_pairs]
        pairs_secondelems = [pair[1] for pair in residue_pairs]
        dm[pair_firstelems, pairs_secondelems] = scores
        dm[pairs_secondelems, pair_firstelems] = scores           
    
    # return the output string and the matrix
    return (outstr, dm)


def calc_potential(distances, \
                   ordered_sparses, \
                   pdb, \
                   uni, \
                   distco, \
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


def load_gmxff(jsonfile):
    _jsonfile = open(jsonfile,'r')
    return json.load(_jsonfile)


def calc_dist_matrix(uni, \
               idxs, \
               chosenselections, \
               co, \
               mindist = False, \
               mindist_mode = None, \
               type1char = "p", \
               type2char = "n"):
    
    numframes = len(uni.trajectory)
    final_percmat = \
        np.zeros((len(chosenselections), len(chosenselections)))
    
    logstr = "Distance matrix will be {:d}x{:d} ({:d} elements)"
    log.info(logstr.format(len(idxs), len(idxs), len(idxs)**2))

    distmats = []

    if mindist:
        P = []
        N = []
        Pidxs = []
        Nidxs = []
        Psizes = []
        Nsizes = []

        for i in range(len(idxs)):
            if idxs[i][3][-1] == type1char:
                P.append(chosenselections[i])
                Pidxs.append(idxs[i])
                Psizes.append(len(chosenselections[i]))
            elif idxs[i][3][-1] == type2char:
                N.append(chosenselections[i])
                Nidxs.append(idxs[i])
                Nsizes.append(len(chosenselections[i]))
            else: 
                errstr = \
                    "Accepted values are either 'n' or 'p', " \
                    "but {:s} was found."
                raise ValueError(errstr.format(idxs[i][3][-1]))

        Nsizes = np.array(Nsizes, dtype = np.int)
        Psizes = np.array(Psizes, dtype = np.int)

	    if mindist_mode == "diff":
            sets = [(P,N)]
            sets_idxs = [(Pidxs, Nidxs)]
            sets_sizes = [(Psizes, Nsizes)]
        elif mindist_mode == "same":
            sets = [(P,P),(N,N)]
            sets_idxs = [(Pidxs,Pidxs), (Nidxs,Nidxs)]
            sets_sizes = [(Psizes,Psizes), (Nsizes,Nsizes)]
        elif mindist_mode == "both":
            sets = [(chosenselections, chosenselections)]
            sets_idxs = [(idxs, idxs)]
            sizes =  [len(s) for s in chosenselections]
            sets_sizes = [(sizes,sizes)]               
        else:
            choices = ["diff", "same", "both"]
            errstr = \
                "Accepted values for mindist_mode are {:s}, " \
                "but you provided {:s}."
            raise ValueError(errstr.format(", ".join(choices), \
                                           mindist_mode))

        percmats = []
        coords = [([[], []]) for s in sets]

        numframe = 1
        for ts in uni.trajectory:
            logstr = \
                "Caching coordinates: frame {:d} / {:d} ({:3.1f}%)\r"
            sys.stdout.write(logstr.format(\
                                numframe, \
                                numframes, \
                                float(numframe)/float(numframes)*100.0))
            sys.stdout.flush()
            numframe += 1
            
            for set_index, s in enumerate(sets):
                if s[0] == s[1]:
                    # triangular case
                    log.info("Caching coordinates...")
                    for group in s[0]:
                        coords[set_index][0].append(group.positions)
                        coords[set_index][1].append(group.positions)
                else:
                    # square case
                    log.info("Caching coordinates...")
                    for group in s[0]:
                        coords[set_index][0].append(group.positions)
                    for group in s[1]:
                        coords[set_index][1].append(group.positions)

        for set_index, s in enumerate(sets):
            # recover the final matrix
            if s[0] == s[1]:
                # triangular case
                this_coords = \
                    np.array(\
                        np.concatenate(coords[si][0]), \
                        dtype = np.float64)

                inner_loop = LoopDistances(this_coords, this_coords, co)
                percmats.append(\
                    inner_loop.run_triangular_mindist(\
                        sets_sizes[set_index][0]))

            else:
                # square case
                this_coords1 = \
                    np.array(\
                        np.concatenate(coords[set_index][0]), \
                        dtype = np.float64)
                
                this_coords2 = \
                    np.array(\
                        np.concatenate(coords[set_index][1]), \
                        dtype = np.float64)
                
                inner_loop = LoopDistances(this_coords1, this_coords2, co)
                percmats.append(\
                    inner_loop.run_square_mindist(\
                        sets_sizes[set_index][0], \
                        sets_sizes[set_index][1]))

        for set_index, s in enumerate(sets): 
            # recover the final matrix
            Pidxs = sets_idxs[si][0]
            Nidxs = sets_idxs[si][1]
            if s[0] == s[1]:
                # triangular case
                for j in range(len(s[0])):
                    for k in range(0,j):
                        final_percmat[idxs.index(Pidxs[j]), \
                                                 idxs.index(Pidxs[k])] = \
                                            percmats[si][j,k]
                        
                        final_percmat[idxs.index(Pidxs[k]), \
                                                 idxs.index(Pidxs[j])] = \
                                            percmats[si][j,k]
            else: 
                # square case
                for j in range(len(s[0])):
                    for k in range(len(s[1])):
                        final_percmat[idxs.index(Pidxs[j]), \
                                                 idxs.index(Nidxs[k])] = \
                                            percmats[si][j,k]
                        
                        final_percmat[idxs.index(Nidxs[k]), \
                                                cidxs.index(Pidxs[j])] = \
                                            percmats[si][j,k]
 
        final_percmat = \
            np.array(final_percmat, dtype = np.float)/numframes*100.0
                     
    else:
        all_coms = []
        numframe = 1
        for ts in uni.trajectory:
            logstr = "Now analyzing: frame {:d} / {:d} ({:3.1f}%)\r"
            sys.stdout.write(logstr.format(\
                                numframe, \
                                numframes, \
                                float(numframe)/float(numframes)*100.0))
            sys.stdout.flush()
            
            numframe += 1
            distmat = \
                np.zeros((len(chosenselections), len(chosenselections)))
            
            # empty matrices of centers of mass
            coms = np.zeros([len(chosenselections),3])
            # fill the matrix
            for j in range(len(chosenselections)):
                coms[j,:] = chosenselections[j].centerOfMass()
            all_coms.append(coms)

        all_coms = np.concatenate(all_coms)
        inner_loop = LoopDistances(all_coms, all_coms, co)
        percmat = inner_loop.run_triangular_calc_dist_matrix(coms.shape[0])
        final_percmat = np.array(percmat, dtype = np.float)/numframes*100.0

    return final_percmat, distmats


def load_sys(pdb, gro, xtc):    
    uni = mda.Universe(gro, xtc)       
    pdb = mda.Universe(pdb)
    return pdb, uni


def assign_ff_masses(ffmasses, idxs, chosenselections):
    ffdata = load_gmxff(ffmasses)
    for i in range(len(idxs)):
        for atom in chosenselections[i]:
            try:                
                atom.mass = ffdata[1][atom.residue.resname][atom.name]
            except:
                warnstr = \
                    "Atom type not recognized (resid {:d}, " \
                    "resname {:s}, atom {:s}). " \
                    "Atomic mass will be guessed."
                log.warning(warnstr.format(atom.residue.resid, \
                                           atom.residue.resname, \
                                           atom.name))   


def generate_custom_identifiers(pdb, uni, **kwargs):
    selstrings = kwargs["selections"]
    names = kwargs["names"]
    if len(names) != len(selstrings):
        errstr = "names and selections must have the same lenght."
        raise ValueError(errstr)
    
    chosenselections = []
    identifiers = []
    idxs = []
    for i in range(len(selstrings)):
        try:
            chosenselections.append(uni.select_atoms(selstrings[i]))
            identifiers.append((names[i], "", "", ""))
            idxs.append((names[i], names[i], "", "", ""))

            logstr = "Selection \"{:s}\" found with {:d} atoms."
            sys.stdout.write(logstr.format(names[i], \
                                           len(chosenselections[-1])))
        except:
            warnstr = \
                "Could not select \"{:s}\". Selection will be skipped."
            log.warning(names[i])
    
    return identifiers, idxs, chosenselections


def generate_cgi_identifiers(pdb, uni, **kwargs):
    cgs = kwargs["cgs"]
    idxs = []
    chosenselections = []
    # preprocess CGs: divide wolves and lambs
    for res, dic in cgs.items(): 
        for cgname, cg in dic.items():
            # True : set of atoms that must exist (negative of negative)
            # False: set atoms that must NOT exist (positive of negative)
            cgs[res][cgname] =  \
                {\
                    True : \
                        set(\
                            filter(\
                                lambda x: not x.startswith("!"),cg)), \
                    False : \
                        set(\
                            [j[1:] for j in filter(\
                                lambda x : x.startswith("!"), cg)]) \
                }
    
    identifiers = \
        [(res.segid, res.resid, res.resname, "") for res in pdb.residues]

    for res in uni.residues:
        segid = res.segid
        resname = res.resname
        resid = res.resid
        setcurnames = set(res.atoms.names)
        try:
            for cgname, cg in cgs[resname].items():
                atoms_to_keep = cg[True]
                condition_to_keep = \
                    atoms_to_keep.issubset(setcurnames) \
                    and not bool(cg[False] & setcurnames)
                if condition_to_keep:
                    idxs.append((segid, resid, resname, cgname))
                    selstring = \
                        "resid {:d} and (name ".format(resid) + \
                        " or name ".join(atoms_to_keep) + ")"
                        
                    chosenselections.append(uni.select_atoms(selstring))

                    log.info(\
                        "{:s} {:s} ({:s})".format(\
                            chosenselections[-1][0].resid, \
                            chosenselections[-1][0].resname, \
                            ", ".join([a.name for a in chosenselections[-1]])))

        except KeyError:
            logstr = \
                "Residue {:s} is not in the charge recognition set. " \
                "Will be skipped."
            log.warn(logstr.format(resname))

    return (identifiers, chosenselections)
 
      
def generate_sci_identifiers(pdb, uni, **kwargs):
    reslist = kwargs["reslist"]
    log.info("Selecting residues: {:s}".format(", ".join(reslist)))
    excluded_atoms = \
        ["CA", "C", "O", "N", "H", "H1", "H2", \
         "H3", "O1", "O2", "OXT", "OT1", "OT2"]
    identifiers=[]
    chosenselections=[]
    idxs=[]
    for i in range(len(pdb.segments)):
        resids = pdb.segments[i].resids()
        identifiers.extend( [ ( pdb.segments[i].name, resids[j], pdb.segments[i].residues[j].name, "sidechain" ) for j in range(len(resids)) ] ) 
    log.info("Chosen selections:")
    for i in reslist:
        idxs.extend( [ identifiers[j] for j in range(len(identifiers)) if identifiers[j][2] == i[0:3] ] )        
        for j in uni.residues:
            if j.name[0:3] == i:
                chosenselections.append( mda.core.AtomGroup.AtomGroup([ k for k in j.atoms if k.name not in excluded_atoms ]) )
                log.info("%s %s (%s)" % (j.resids()[0], j.name, ", ".join([ k.name for k in j.atoms if k.name not in excluded_atoms ])))
    return identifiers,idxs,chosenselections
    
     
def calc_sc_fullmatrix(identifiers, idxs, percmat, perco):
    fullmatrix = np.zeros((len(identifiers),len(identifiers)))
    for i in range(len(identifiers)):
        for j in range(0,i):
            if identifiers[i] in idxs and identifiers[j] in idxs:
                fullmatrix[i,j] = percmat[idxs.index(identifiers[i]), idxs.index(identifiers[j])]
                fullmatrix[j,i] = percmat[idxs.index(identifiers[i]), idxs.index(identifiers[j])]
    return fullmatrix

def calc_cg_fullmatrix(identifiers, idxs, percmat, perco):
    fullmatrix = np.zeros((len(identifiers),len(identifiers)))
    lastsize=0
    stopsize=0  
    clashes=[]
    origidxs=list(idxs)
    assert percmat.shape[0] == percmat.shape[1] ## this better be true, or we really fucked up 
    assert percmat.shape[0] != lastsize         ## avoid infinite looping in case something goes wrong
    lastsize = percmat.shape[0]
    stopsize = 0
    corrected_percmat = percmat.copy()
    i=0
    while i < percmat.shape[0]:
        rescgs = map(idxs.index,filter(lambda x: x[0:3] == idxs[i][0:3], idxs))
        i+=len(rescgs)
        clashes.append(frozenset(rescgs)) # possibly clashing residues
    corrected_percmat = np.zeros( (len(clashes),len(clashes)) )
    for i in range(len(clashes)):
        for j in range(i):
            thisclash = 0.0
            for k in list(clashes[i]):
                for l in list(clashes[j]):
                    if percmat[k][l] > thisclash:
                        thisclash = percmat[k][l]
            corrected_percmat[i,j] = thisclash
    corrected_idxs = [ idxs[list(i)[0]][0:3]+('',) for i in list(clashes)]
    for i in range(len(identifiers)):
        for j in range(0,i):
            if identifiers[i] in corrected_idxs and identifiers[j] in corrected_idxs:
                fullmatrix[i,j] = corrected_percmat[corrected_idxs.index(identifiers[i]), corrected_idxs.index(identifiers[j])]
                fullmatrix[j,i] = corrected_percmat[corrected_idxs.index(identifiers[i]), corrected_idxs.index(identifiers[j])]
    
    return fullmatrix
    
def do_interact(identfunc, grof=None, xtcf=None ,pdbf=None, pdb=None, uni=None, co=5.0, perco=0.0, ffmasses=None, fullmatrix=None, mindist=False, mindist_mode=None, **identargs ):

    outstr = ""

    idxs = []
    chosenselections = []
    identifiers = []
    
    if not pdb or not uni:
        if not pdbf or not grof or not xtcf:
            raise ValueError
        pdb,uni = load_sys(pdbf,grof,xtcf)
    
    identifiers,idxs,chosenselections = identfunc(pdb,uni,**identargs)

    if ffmasses is not None:
        try:
            assign_ff_masses(ffmasses,idxs,chosenselections)
        except IOError as (errno, strerror):
            log.warning("force field file not found or not readable. Masses will be guessed.")
            pass
    else:
        log.info("No force field assigned: masses will be guessed.")

    percmat,distmats = calc_dist_matrix(uni, idxs,chosenselections, co, mindist=mindist, mindist_mode=mindist_mode)

    short_idxs = [ i[0:3] for i in idxs ]
    short_identifiers = [ i[0:3] for i in identifiers ]

    for i in range(len(percmat)):
        for j in range(0,i):
            if percmat[i,j] > perco:
                this1 = short_identifiers[short_identifiers.index(short_idxs[i])]
                this2 = short_identifiers[short_identifiers.index(short_idxs[j])]
                outstr+= "%s-%d%s_%s:%s-%d%s_%s\t%3.1f\n" % (this1[0], this1[1], this1[2], idxs[i][3], this2[0], this2[1], this2[2], idxs[j][3], percmat[i,j])

    if fullmatrix:
        return (outstr, fullmatrix(identifiers, idxs, percmat, perco))

    return (outstr, None)

def do_hbonds(sel1, sel2, grof=None ,xtcf=None, pdbf=None, pdb=None, uni=None, update_selection1=True, update_selection2=True, filter_first=False, distance=3.0, angle=120, perco=0.0, perresidue=False, dofullmatrix=False, other_hbs=None):
    
    outstr=""
    
    from MDAnalysis.analysis import hbonds
    
    if not pdb or not uni:
        if not pdbf or not grof or not xtcf:
            raise ValueError
        pdb,uni = load_sys(pdbf,grof,xtcf)


    hb_basenum = 1
    
    try:
        sel1atoms = uni.selectAtoms(sel1)
    except:
        log.error("ERROR: selection 1 is invalid")
    try:
        sel2atoms = uni.selectAtoms(sel2)
    except:
        log.error("ERROR: selection 2 is invalid")
        
    
    if other_hbs:  # custom names
        class Custom_HydrogenBondAnalysis(hbonds.HydrogenBondAnalysis):
            DEFAULT_DONORS = {"customFF":other_hbs['DONORS']}
            DEFAULT_ACCEPTORS = {"customFF":other_hbs['ACCEPTORS']}
        hb_ff = "customFF"
    else:
        class Custom_HydrogenBondAnalysis(hbonds.HydrogenBondAnalysis):
            pass
        hb_ff = "CHARMM27"
    
    h = Custom_HydrogenBondAnalysis(uni, sel1, sel2, distance=distance, angle=angle, forcefield=hb_ff)
    
    log.info("Will use acceptors: %s" %  ", ".join(h.DEFAULT_ACCEPTORS[hb_ff]))
    log.info("Will use donors:    %s" %  ", ".join(h.DEFAULT_DONORS[hb_ff]))
    log.info("Running hydrogen bonds analysis . . .")
    
    h.run()
    log.info("Done! Finalizing . . .")
    data = h.timeseries

    identifiers = []
    pdbresidues=[]
    uniresidues=[]
    uni_identifiers = []
    fullmatrix = None
    
    for i in range(len(uni.segments)): #build global identifiers
        resids = uni.segments[i].resids()
        uniresidues.extend( [j for j in uni.segments[i]])
        uni_identifiers.extend( [ ( uni.segments[i].name, resids[j], uni.segments[i].residues[j].name, "residue" ) for j in range(len(resids)) ] )
        
    for i in range(len(pdb.segments)): #build global identifiers
        resids = pdb.segments[i].resids()
        pdbresidues.extend([j for j in pdb.segments[i]])
        identifiers.extend( [ ( pdb.segments[i].name, resids[j], pdb.segments[i].residues[j].name, "residue" ) for j in range(len(resids)) ] ) 

    if perresidue or dofullmatrix:
        outdata={}
        setsetlist = []
        
        setlist = list(set(zip(*zip(*list(chain(*data)))[0:2])))
        identifiers_setlist = list(set( [ frozenset(((uni.atoms[hbond[0]-1].segment.name, uni.atoms[hbond[0]-1].resid, uni.atoms[hbond[0]-1].residue.name, "residue"), (uni.atoms[hbond[1]-1].segment.name, uni.atoms[hbond[1]-1].resid, uni.atoms[hbond[1]-1].residue.name, "residue"))) for hbond in setlist ] ))

        for i in identifiers_setlist:
            outdata[i] = 0
        for frame in data:
            thisframe = set([ frozenset(((uni.atoms[hbond[0]-1].segment.name, uni.atoms[hbond[0]-1].resid, uni.atoms[hbond[0]-1].residue.name, "residue"), (uni.atoms[hbond[1]-1].segment.name, uni.atoms[hbond[1]-1].resid, uni.atoms[hbond[1]-1].residue.name, "residue"))) for hbond in frame])
            for resc in thisframe:
                outdata[resc] +=1
        if perresidue:
            outdata2 = outdata.copy()
            for k in outdata2.keys():
                outdata2[k] = float(outdata2[k])/float(uni.trajectory.numframes)*100
                if outdata2[k] > perco:
                    if len(k) == 2:
                        outstr+= "%s:%s\t\t%3.2f\n" % ( list(k)[0][0]+list(k)[0][1], list(k)[1][0]+list(k)[1][1], outdata[k])
                    elif len(k) == 1:
                        outstr+= "%s:%s\t\t%3.2f\n" % ( list(k)[0],list(k)[0], outdata[k])

        fullmatrix = None
        if dofullmatrix:  
            fullmatrix = np.zeros((len(identifiers),len(identifiers)))
            for hb,val in outdata.iteritems():
                hbond = list(hb)
                m = uni_identifiers.index(hbond[0])
                if len(hbond) == 1:
                    n = m
                else:
                    n = uni_identifiers.index(hbond[1])

                fullmatrix[m,n] = val
                fullmatrix[n,m] = val
            
            fullmatrix = fullmatrix/float(uni.trajectory.numframes)*100.0 

    if not perresidue:
       
        table = h.count_by_type()
        hbonds_identifiers = []
        for hbond in table:            
            hbonds_identifiers.append( ((uni.atoms[hbond[0]-1].segment.name, 
                                    uni.atoms[hbond[0]-1].residue.id, 
                                    uni.atoms[hbond[0]-1].residue.name, "residue"), 
                                   (uni.atoms[hbond[1]-1].segment.name, 
                                    uni.atoms[hbond[1]-1].residue.id, 
                                    uni.atoms[hbond[1]-1].residue.name, "residue")) )

        for i,hbond in enumerate(hbonds_identifiers):
            if table[i][-1]*100.0 > perco:
                atom1 = identifiers[uni_identifiers.index(hbond[0])]
                atom2 = identifiers[uni_identifiers.index(hbond[1])]
 
                outstr += "%s-%d%s_%s:%s-%d%s_%s\t\t%3.2f\n" % ( atom1[0], atom1[1], atom1[2], table[i][4], atom2[0], atom2[1], atom2[2], table[i][8], table[i][-1]*100.0 )

    return (outstr, fullmatrix)
    
    

