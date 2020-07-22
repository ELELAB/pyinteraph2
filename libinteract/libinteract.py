#!/usr/bin/env python
# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-

#    PyInteraph, a software suite to analyze interactions and 
#    interaction network in structural ensembles.
#    Copyright (C) 2013 Matteo Tiberti <matteo.tiberti@gmail.com>, 
#                       Gaetano Invernizzi, Yuval Inbar, 
#                       Matteo Lambrughi, Gideon Schreiber, 
#                       Elena Papaleo <elena.papaleo@unimib.it> 
#                                     <elena.papaleo@bio.ku.dk>
#
#    This program is free software: you can redistribute it 
#    and/or modify it under the terms of the GNU General Public 
#    License as published by the Free Software Foundation, either 
#    version 3 of the License, or (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  
#    If not, see <http://www.gnu.org/licenses/>.

from __future__ import absolute_import

import collections
import configparser as cp
import itertools
import json
import logging as log
import struct
import sys

from innerloops import LoopDistances
import numpy as np
import MDAnalysis as mda
from MDAnalysis.analysis.distances import distance_array


############################## PARSING ##############################

class Sparse:
    def __repr__(self):
        fmtrepr = \
            "<Sparse r1={:d} ({:d},{:d}), r2={:d} ({:d},{:d}), {:d} bins>"
        
        return fmtrepr.format(self.r1, self.p1_1, self.p1_2, self.r2, \
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
        if i != 0:
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
        expected = len(data)
        errstr = \
            f"Could not finish parsing the file " \
            f"{potential_file} ({pointer} bytes read, " \
            f"{expected} expected)."
        raise ValueError(errstr)

    sparses_dict = {}
    for i in range(len(residues_list)):
        sparses_dict[residues_list[i]] = {}
        for j in range(i):            
            sparses_dict[residues_list[i]][residues_list[j]] = {}
            
    for s in sparses:
        if s:
            sparses_dict[residues_list[s[0].r1]][residues_list[s[0].r2]] = s[0]
    
    sys.stodout.write(f"Done parsing file {potential_file}!")
    
    return sparses_dict


def parse_kbpatomsfile(kbpatomsfile):
    """Parse the file containing the list of atoms
    for potential calculation."""

    # try to open the file
    try:
        fh = open(kbpatomsfile)
    except:
        raise IOError(f"Could not open file {kbpatomsfile}.")
    # parse the file
    with fh:
        data = {}
        for line in fh:
            res, atoms = line.strip().split(":")
            data[res.strip()] = [a.strip() for a in atoms.split(",")]
        return data  


def parse_cgsfile(cgsfile):
    """Parse the file containing the definition of
    charged groups."""

    grpstr = "CHARGED_GROUPS"
    resstr = "RESIDUES"
    defgrpstr = "default_charged_groups" 
    # create the config parser and read the config file
    cfg = cp.ConfigParser()
    # try to read the file   
    try:
        cfg.read(cgsfile)
    except:
        logstr = \
            f"File {cgsfile} not readeable or not in the right format."
        raise IOError(logstr)

    # charged groups
    cggroups = cfg.options(grpstr)
    cggroups.remove(defgrpstr)
    cggroups = [item.strip() for item in cggroups]
    # default charged groups
    defcg = cfg.get(grpstr, defgrpstr).split(",")
    defcg = [item.strip() for item in defcg]
    # residues
    residues = cfg.options(resstr)
    # group definitions
    groupdef = \
        {g : [i.strip() for i in cfg.get(grpstr, g).split(",")] \
         for g in (cggroups + defcg)}
    # empty dictionary to store the output
    out = {}
    for res in residues:
        # convert the residue name into all uppercase
        res = res.upper()
        # default charged groups
        defcgdict = {cg : groupdef[cg] for cg in defcg}
        # other charged groups for this residue
        othercgs = cfg.get(resstr, res).split(",")
        othercgdict = {i : groupdef[i.lower()] for i in othercgs if i != ""}
        # merge the two dictionaries
        out[res] = {**defcgdict, **othercgdict}
    # return the final dictionary
    return out


def parse_hbsfile(hbsfile):
    """Parse the file containing the definition of donors
    and acceptors to analyze hydrogen bonds.
    """
    
    hbsstr = "HYDROGEN_BONDS"
    acceptorsstr = "ACCEPTORS"
    donorsstr = "DONORS"
    # create the config parser and read the config file
    cfg = cp.ConfigParser()
    try:
        cfg.read(hbsfile)
    except:
        errstr = \
            f"File {hbsfile} not readeable or not in the right format."
        raise IOError(errstr)
    # get hydrogen bond acceptors
    acceptors = cfg.get(hbsstr, acceptorsstr).strip().split(",")
    acceptors = [i.strip() for i in acceptors]
    # get hydrogen bond donors
    donors = cfg.get(hbsstr, donorsstr).strip().split(",")
    donors = [i.strip() for i in donors]
    # return dictionary of donors and acceptors
    return dict(ACCEPTORS = acceptors, DONORS = donors)



############################## POTENTIAL ##############################

def calc_potential(distances, \
                   orderedsparses, \
                   refuni, \
                   uni, \
                   kbT):
    """Calculate statistical potential."""

    generalcutoff = 5.0  
    tot = 0
    done = 0
    thisdist = np.zeros((2,2), dtype = np.float64)
    scores = np.zeros((distances.shape[1]), dtype = np.float64)  
    for frame in distances:
        for pn, thisdist in enumerate(frame):
            if np.any(thisdist < generalcutoff):
                searchstring = \
                    "".join(\
                        [chr(int(d * orderedsparses[pn].step + 1.5)) \
                         for d in thisdist])
                try:
                    probs = - kbT * orderedsparses[pn].bins[searchstring]
                except KeyError:
                    probs = 0.0
                
                scores[pn] += probs
            else:
                scores[pn] += 0.0
    
    return scores/distances.shape[0]


def do_potential(kbpatoms, \
                 reslist, \
                 potentialfile, \
                 uni, \
                 refuni, \
                 dofullmatrix, \
                 kbT, \
                 seqdistco, \
                 parse_sparse_func = parse_sparse, \
                 calc_potential_func = calc_potential):
    """Perform potential analysis."""

    log.info("Loading potential definition . . .")
    sparses = parse_sparse_func(potentialfile, reslist)

    okres = []
    discardedres = set()
    respairs = []
    atomsel = []
    orderedsparses = []
    numframes = len(uni.trajectory)

    for res in uni.residues:
        # check if the residue type is one of
        # those included in the list
        if res.resname in reslist:
            # add it to the accepted residues
            okres.append(res)
        else:
            # add it to the discarded residues
            # and continue
            discardedres.add(res)
            continue
        # for each other accepted residue
        for res2 in okres[:-1]:
            res1 = res 
            seqdist = abs(res1.ix - res2.ix)
            res1segid = res1.segment.segid
            res2segid = res2.segment.segid
            if not (seqdist < seqdistco or res1segid != res2segid):
                # string comparison ?!
                if res2.resname < res1.resname:
                    ii, j = j, ii
                
                thissparse = sparses[res1.resname][res2.resname]
                # get the four atoms for the potential
                atom0, atom1, atom2, atom3 = \
                    (kbpatoms[res1.resname][thissparse.p1_1],
                     kbpatoms[res1.resname][thissparse.p1_2],
                     kbpatoms[res2.resname][thissparse.p2_1],
                     kbpatoms[res2.resname][thissparse.p2_2])         
                try:
                    atom0ix = res1.atoms.names.tolist().index(atom0)
                    atom1ix = res1.atoms.names.tolist().index(atom1)
                    atom2ix = res2.atoms.names.tolist().index(atom2)
                    atom3ix = res2.atoms.named.tolist().index(atom3)
                    selectedatoms = mda.core.groups.AtomGroup(\
                        res1.atoms[atom0ix], res1.atoms[atom1ix], \
                        res2.atoms[atom2ix], res2.atoms[atom3ix])
                except:
                    # inform the user about the problem and continue
                    warnstr = "Could not identify essential atoms " \
                              "for the analysis ({:s}{:d}, {:s}{:d})"
                    log.warning(\
                        warnstr.format(res1.resname, res1.resid, \
                                       res2.resname, res2.resid))                
                    continue
                
                respairs.append((res1, res2))
                atomsel.append(selectedatoms)
                orderedsparses.append(thissparse)

    # create an matrix of floats to store scores (initially
    # filled with zeros)
    scores = np.zeros((len(respairs)), dtype = np.float64)
    # set coordinates to None
    coords = None
    # for each frame in the trajectory
    numframe = 1
    for ts in uni.trajectory:
        # log the progress along the trajectory
        logstr = "Now analyzing: frame {:d} / {:d} ({:3.1f}%)\r"
        sys.stdout.write(\
            logstr.format(numframe, \
                          numframes, \
                          float(numframe)/float(numframes)*100.0))
        sys.stdout.flush()
        # update the frame number
        numframe += 1    
        
        # create an array of coordinates by concatenating the arrays of
        # atom positions in the selections row-wise
        coords = \
            np.array(\
                np.concatenate([sel.positions for sel in atomsel]), \
                dtype = np.float64)

        inner_loop = LoopDistances(coords, coords, None)
        # compute distances
        distances = inner_loop.run_potential_distances(len(atomsel), 4, 1)
        # compute scores
        scores += \
            calc_potential_func(distances = distances, \
                                orderedsparses = orderedsparses, \
                                refuni = refuni, \
                                uni = uni, \
                                kbT = kbT)
    
    # divide the scores for the number of frames
    scores /= float(numframes)
    # create the output string
    outstr = ""
    # set the format for the representation of each pair of
    # residues in the output string
    outstrfmt = "{:s}-{:s}{:s}:{:s}-{:s}{:s}\t{:.3f}\n"
    for i, score in enumerate(scores):
        if abs(score) > 0.000001:
            # update the output string
            outstr +=  \
                outstrfmt.format(\
                    refuni.residues[respairs[i][0]].segment.segid, \
                    refuni.residues[respairs[i][0]].resname, \
                    refuni.residues[respairs[i][0]].resid, \
                    refuni.residues[respairs[i][1]].segment.segid, \
                    refuni.residues[respairs[i][1]].resname, \
                    refuni.residues[respairs[i][1]].resid, \
                    score)  
    # inizialize the matrix to None  
    fullmatrix = None   
    if dofullmatrix:
        # if requested, create the matrix
        fullmatrix = np.zeros((len(refuni.residues), len(refuni.residues)))
        # use numpy "fancy indexing" to fill the matrix
        # with scores at the corresponding residues pairs
        # positions
        firstelems = [pair[0] for pair in respairs]
        secondelems = [pair[1] for pair in respairs]
        fullmatrix[firstelems, secondelems] = scores
        fullmatrix[secondelems, firstelems] = scores
    # return the output string and the matrix
    return (outstr, fullmatrix)



###################### SALT BRIDGES and CONTACTS ######################

def calc_dist_matrix(uni, \
                     idxs, \
                     selections, \
                     co, \
                     sb = False, \
                     sbmode = "diff", \
                     poschar = "p", \
                     negchar = "n"):
    """Compute matrix of distances (salt bridges or
    generic residue-residue distances (e.g. hydrophobic
    contacts)."""
    
    numframes = len(uni.trajectory)
    # initialize the final matrix
    percmat = np.zeros((len(selections), len(selections)), \
                       dtype = np.float64)
    # if salt bridges
    if sb:
        # lists for positively charged atoms
        pos = []
        posidxs = []
        possizes = []
        # lists for negatively charged atoms
        neg = []
        negidxs = []
        negsizes = []
        # for each ix
        for i in range(len(idxs)):
            # character in the index indicating the
            # charge of the atom
            chargechar = idxs[i][3][-1]
            # if the index contains the indication of a
            # positively charged atom
            if chargechar == poschar:
                pos.append(selections[i])
                posidxs.append(idxs[i])
                possizes.append(len(selections[i]))
            # if the index contains the indication of a
            # negatively charged atom
            elif chargechar == negchar:
                neg.append(selections[i])
                negidxs.append(idxs[i])
                negsizes.append(len(selections[i]))
            # if none of the above
            else: 
                errstr = f"Accepted values are either '{poschar}' " \
                         f"or '{negchar}', but {chargechar} was found."
                raise ValueError(errstr)

        # convert lists of positions into arrays
        possizes = np.array(possizes, dtype = np.int)
        negsizes = np.array(negsizes, dtype = np.int)
        # if we are interested in interactions between atoms
        # with different charges
        if sbmode == "diff":
            sets = [(pos, neg)]
            setsidxs = [(posidxs, negidxs)]
            setssizes = [(possizes, negsizes)]
        # if we are interested in interactions between atoms
        # with the same charge
        elif sbmode == "same":
            sets = [(pos, pos), (neg, neg)]
            setsidxs = [(posidxs, posidxs), (negidxs, negidxs)]
            setssizes = [(possizes, possizes), (negidxs, negsizes)]
        # if we are interested in both
        elif sbmode == "both":
            sets = [(selections, selections)]
            setsidxs = [(idxs, idxs)]
            sizes =  [len(s) for s in selections]
            setssizes = [(sizes, sizes)]
        # unrecognized choice             
        else:
            choices = ["diff", "same", "both"]
            errstr = \
                "Accepted values for 'sbmode' are {:s}, " \
                "but {:s} was found."
            raise ValueError(errstr.format(", ".join(choices), sbmode))

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
            for sindex, s in enumerate(sets):
                if s[0] == s[1]:
                    # triangular case
                    for group in s[0]:
                        coords[sindex][0].append(group.positions)
                        coords[sindex][1].append(group.positions)
                else:
                    # square case
                    for group in s[0]:
                        coords[sindex][0].append(group.positions)
                    for group in s[1]:
                        coords[sindex][1].append(group.positions)

        for sindex, s in enumerate(sets):
            # recover the final matrix
            if s[0] == s[1]:
                # triangular case
                thiscoords = \
                    np.array(np.concatenate(coords[sindex][0]), \
                             dtype = np.float64)
                # compute the distances within the cut-off
                innerloop = LoopDistances(thiscoords, thiscoords, co)
                percmats.append(innerloop.run_triangular_mindist(\
                                setssizes[sindex][0]))
            else:
                # square case
                thiscoords1 = \
                    np.array(np.concatenate(coords[sindex][0]), \
                             dtype = np.float64)              
                thiscoords2 = \
                    np.array(np.concatenate(coords[sindex][1]), \
                             dtype = np.float64)
                # compute the distances within the cut-off
                innerloop = LoopDistances(thiscoords1, thiscoords2, co)
                percmats.append(innerloop.run_square_mindist(\
                                setssizes[sindex][0], \
                                setssizes[sindex][1]))

        for sindex, s in enumerate(sets): 
            # recover the final matrix
            posidxs = setsidxs[sindex][0]
            negidxs = setsidxs[sindex][1]
            if s[0] == s[1]:
                # triangular case
                for j in range(len(s[0])):
                    for k in range(0, j):
                        ix_j = idxs.index(posidxs[j])
                        ix_k = idxs.index(posidxs[k])
                        percmat[ix_j, ix_k] = percmats[sindex][j,k]         
                        percmat[ix_k, ix_j] = percmats[sindex][j,k]
            else: 
                # square case
                for j in range(len(s[0])):
                    for k in range(len(s[1])):
                        ix_j_p = idxs.index(posidxs[j])
                        ix_k_n = idxs.index(negidxs[k])
                        percmat[ix_j_p, ix_k_n] = percmats[sindex][j,k]         
                        percmat[ix_k_n, ix_j_p] = percmats[sindex][j,k]
                     
    else:
        # empty list of matrices of centers of mass
        allcoms = []
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
            comslist = [sel.center(sel.masses) for sel in selections]
            allcoms.append(np.array(comslist, dtype = np.float64))
        # create a matrix of all centers of mass along the trajectory
        allcoms = np.concatenate(allcoms)
        # compute the distances within the cut-off
        innerloop = LoopDistances(allcoms, allcoms, co)
        percmat = innerloop.run_triangular_calc_dist_matrix(coms.shape[0])
    # convert the matrix into an array
    percmat = np.array(percmat, dtype = np.float64)/numframes*100.0
    # return the matrix
    return percmat


def assign_ff_masses(ffmasses, selections):
    """Assign masses to atoms based on the force field used."""

    # load force field data
    ffdata = json.load(open(ffmasses), "r")
    for selection in selections:
        # for each atom
        for atom in selection:
            atomresname = atom.residue.resname
            atomresid = atom.residue.resid
            atomname = atom.name
            try:                
                atom.mass = ffdata[1][atomresname][atomname]
            except:
                warnstr = \
                    f"Atom type not recognized (resid {atomresid}, " \
                    f"resname {atomresname}, atom {atomname}). " \
                    f"Atomic mass will be guessed."
                log.warning(warnstr)   

# commented out since never used

"""
def generate_custom_identifiers(refuni, uni, **kwargs):

    # get selection strings
    selstrings = kwargs["selections"]
    # get names
    names = kwargs["names"]
    # raise an error if their lengths do not match
    if len(names) != len(selstrings):
        errstr = "'names' and 'selections' must have the same lenght."
        raise ValueError(errstr)
    
    selections = []
    identifiers = []
    idxs = []
    for selstring, name in zip(selstrings, names):
        try:
            selections.append(uni.select_atoms(selstring))
        except:
            warnstr = \
                "Could not select '{:s}'. Selection will be skipped."
            log.warning(name)
            continue
        
        identifiers.append((name, "", "", ""))
        idxs.append((name, name, "", "", ""))
        # log the selection
        logstr = "Selection '{:s}' found with {:d} atoms."
        sys.stdout.write(logstr.format(name, len(selections[-1])))
    
    return identifiers, idxs, selections
"""

def generate_cg_identifiers(refuni, uni, **kwargs):
    """Generate charged atoms identifiers."""

    cgs = kwargs["cgs"]
    # preprocess CGs: divide wolves and lambs
    filterfunc = lambda x: not x.startswith("!")
    for res, dic in cgs.items(): 
        for cgname, cg in dic.items():
            # True : atoms that must exist (negative of negative)
            trueset = set(filter(filterfunc, cg))
            # False: atoms that must NOT exist (positive of negative)
            falseset = set([j[1:] for j in filter(filterfunc, cg)])
            # update CGs
            cgs[res][cgname] = {True : trueset, False : falseset}
    
    # list of identifiers
    identifiers = [(r.segid, r.resid, r.resname, "") for r in refuni.residues]
    # empty lists of IDs and atom selections
    idxs = []
    selections = []
    # atom selection string
    selstring = "segid {:s} and resid {:d} and (name {:s})"
    # for each residue in the Universe
    for res in uni.residues:
        segid = res.segid
        resname = res.resname
        resid = res.resid
        try:
            cgsitems = cgs[resname].items()
        except KeyError:
            logstr = \
                f"Residue {resname} is not in the charge recognition " \
                f"set. It will be skipped."
            log.warn(logstr)
            continue
        # current atom names
        setcurnames = set(res.atoms.names)
        # for each charged group
        for cgname, cg in cgsitems:
            # get the set of atoms to be kept
            atoms_mustexist = cg[True]
            atoms_mustnotexist = cg[False]
            # set the condition to keep atoms in the current
            # residue, i.e. the atoms that must be present are
            # present and those which must not be present are not
            condtokeep = \
                atoms_mustexist.issubset(setcurnames) and \
                atoms_mustnotexist.isdisjoint(setcurnames)
            # if the condition is met
            if condtokeep:
                idx = (segid, resid, resname, cgname)
                atomstr = " or name ".join(atoms_mustexist)
                selection = uni.select_atoms(selstring.format(\
                                segid, resid, atomstr))                
                # update lists of IDs and atom selections
                idxs.append(idx)
                selections.append(selection)
                # log the selection
                atomnamestr = ", ".join([a.name for a in selection])
                log.info(f"{resid} {resname} ({atomnamestr})")
    # return identifiers, IDs and atom selections
    return (identifiers, idxs, selections)

      
def generate_sc_identifiers(refuni, uni, **kwargs):
    """Generate side chain identifiers."""

    # get the residue names list
    reslist = kwargs["reslist"]
    # log the list of residue names
    log.info("Selecting residues: {:s}".format(", ".join(reslist)))
    # backbone atoms must be excluded
    bbatoms = ["CA", "C", "O", "N", "H", "H1", "H2", \
               "H3", "O1", "O2", "OXT", "OT1", "OT2"]
    # create list of identifiers
    identifiers = \
        [(r.segid, r.resid, r.resname, "sidechain") for r in refuni.residues]
    # create empty lists for IDs and atom selections
    selections = []
    idxs = []
    # atom selection string
    selstr = "segid {:s} and resid {:d} and (name {:s})"
    # start logging the chosen selections
    log.info("Chosen selections:")
    # for each residue name in the residue list
    for resnameinlist in reslist:
        resname3letters = resnameinlist[0:3]
        # update the list of IDs with all those residues matching
        # the current residue type
        for identifier in identifiers:
            if identifier[2] == resname3letters:
                idxs.append(identifier)
        # for each residue in the Universe
        for res in uni.residues:
            if res.resname[0:3] == resnameinlist:
                resid = res.resid
                resname = res.resname
                segid = res.segid
                # get side chain atom names
                scnames = [a.name for a in res.atoms if a.name not in bbatoms]
                scstr = " or name ".join(scnames)
                scnamestr = ", ".join(scnames)
                # get the side chain atom selection
                selection = \
                    uni.select_atoms(selstr.format(segid, resid, scstr))
                # save the selection
                selections.append(selection)
                # log the selection
                log.info(f"{resid} {resname} ({scnamestr})")
    # return identifiers, indexes and selections
    return (identifiers, idxs, selections)
    
     
def calc_sc_fullmatrix(identifiers, idxs, percmat, perco):
    """Calculate side chain-side chain interaction matrix
    (hydrophobic contacts)."""

    # create a matrix of size identifiers x identifiers
    fullmatrix = np.zeros((len(identifiers), len(identifiers)))
    # get where (index) the elements of idxs are in identifiers
    where_idxs_in_identifiers = \
        [identifiers.index(item) for item in idxs]
    # get where (index) each element of idxs is in idxs
    where_idxs_in_idxs = [i for i, item in enumerate(idxs)]
    # get where (i,j coordinates) each element of idxs is in
    # fullmatrix
    pos_identifiers_in_fullmatrix = \
        itertools.combinations(where_idxs_in_identifiers, 2)
    # get where (i,j coordinates) each element of idxs is in
    # percmat (which has dimensions len(idxs) x len(idxs))
    pos_idxs_in_percmat = \
        itertools.combinations(where_idxs_in_idxs, 2)
    # unpack all pairs of i,j coordinates in lists of i 
    # indexes and j indexes
    ifullmatrix, jfullmatrix = zip(*pos_identifiers_in_fullmatrix)
    ipercmat, jpercmat = zip(*pos_idxs_in_percmat)
    # use numpy "fancy indexing" to fill fullmatrix with the
    # values in percmat corresponding to each pair of elements
    fullmatrix[ifullmatrix, jfullmatrix] = percmat[ipercmat, jpercmat]
    fullmatrix[jfullmatrix, ifullmatrix] = percmat[ipercmat, jpercmat]
    # return the full matrix (square matrix)
    return fullmatrix


def calc_cg_fullmatrix(identifiers, idxs, percmat, perco):
    """Calculate charged atoms interaction matrix (salt bridges)."""
    
    # search for residues with duplicate ID
    duplicates = []
    idxindex = 0
    while idxindex < percmat.shape[0]:
        # function to retrieve only residues whose ID (segment ID,
        # residue ID and residue name) perfectly matches the one of
        # the residue currently under evaluation (ID duplication)
        filterfunc = lambda x: x[0:3] == idxs[idxindex][0:3]
        # get where (indexes) residues with the same ID as that
        # currently evaluated are
        rescgs = list(map(idxs.index, filter(filterfunc, idxs)))
        # save the indexes of the duplicate residues
        duplicates.append(frozenset(rescgs))
        # update the counter
        idxindex += len(rescgs)
    # if no duplicates are found, the corrected matrix will have
    # the same size as the original matrix 
    corrpercmat = np.zeros((len(duplicates), len(duplicates)))
    # generate all the possible combinations of the sets of 
    # duplicates (each set represents a residue who found
    # multiple times in percmat)
    duplcomb = itertools.combinations(duplicates)
    # for each pair of sets of duplicates
    for duplresi, duplresj in duplcomb:
        # get where residue i and residue j should be uniquely
        # represented in the corrected matrix
        corrix_i = duplicates.index(duplresi)
        corrix_j = duplicates.index(duplresj)
        # get the positions of all interactions made by each instance
        # of residue i with each instance of residue j in percmat
        ix_i, ix_j = zip(*itertools.product(duplresi, duplresj))
        # use numpy "fancy indexing" to put in corrected_percmat
        # only the strongest interaction found between instances of
        # residue i and instances of residue j
        corrpercmat[corrix_i, corrix_j] = np.max(percmat[ix_i, ix_j])
    # to generate the new IDs, get the first instance of each residue
    # (duplicates all share the same ID) and add an empty string as
    # last element of the ID
    corridxs = [idxs[list(i)[0]][0:3] + ("",) for i in list(duplicates)]
    # create a matrix of size identifiers x identifiers
    fullmatrix = np.zeros((len(identifiers), len(identifiers)))
    # get where (index) the elements of corrected_idxs are in identifiers
    where_idxs_in_identifiers = \
        [identifiers.index(item) for item in corridxs]
    # get where (index) each element of corrected_idxs 
    # is in corrected_idxs
    where_idxs_in_idxs = [i for i, item in enumerate(corridxs)]
    # get where (i,j coordinates) each element of corrected_idxs
    # is in fullmatrix
    pos_identifiers_in_fullmatrix = \
        itertools.combinations(where_idxs_in_identifiers, 2)
    # get where (i,j coordinates) each element of corrected_idxs
    # is in corrected_percmat
    pos_idxs_in_corrpercmat = \
        itertools.combinations(where_idxs_in_idxs, 2)
    # unpack all pairs of i,j coordinates in lists of i 
    # indexes and j indexes
    i_fullmatrix, j_fullmatrix = zip(*pos_identifiers_in_fullmatrix)
    i_corrpercmat, j_corrpercmat = zip(*pos_idxs_in_corrpercmat)
    # use numpy "fancy indexing" to fill fullmatrix with the
    # values in percmat corresponding to each pair of elements
    fullmatrix[i_fullmatrix, j_fullmatrix] = \
        corrpercmat[i_corrpercmat, j_corrpercmat]
    fullmatrix[j_fullmatrix, i_fullmatrix] = \
        corrpercmat[i_corrpercmat, j_corrpercmat]
    # return the full matrix (square matrix)
    return fullmatrix


def do_interact(generate_identifiers_func, \
                refuni, \
                uni, \
                co, \
                perco, \
                ffmasses = None, \
                calc_fullmatrix_func = None, \
                sb = False, \
                sbmode = None, \
                assign_ff_masses_func = assign_ff_masses, \
                calc_dist_matrix_func = calc_dist_matrix, \
                **generate_identifiers_args):
    
    # get identifiers, indexes and atom selections
    identifiers, idxs, selections = \
        generate_identifiers_func(refuni, uni, **generate_identifiers_args)
    # assign atomic masses to atomic selections if not provided
    if ffmasses is None:
        log.info("No force field assigned: masses will be guessed.")
    else:
        try:
            assign_ff_masses_func(ffmasses, selections)
        except IOError:
            logstr = "Force field file not found or not readable. " \
                     "Masses will be guessed."
            log.warning(logstr)     
    # calculate the matrix of persistences
    percmat = calc_dist_matrix_func(uni = uni, \
                                    idxs = idxs,\
                                    selections = selections, \
                                    co = co, \
                                    sb = sb, \
                                    sbmode = sbmode)
    # get shortened indexes and identifiers
    shortidxs = [i[0:3] for i in idxs]
    shortids = [i[0:3] for i in identifiers]
    # set output string and output string format
    outstr = ""
    outstrfmt = "{:s}-{:d}{:s}_{:s}:{:s}-{:d}{:s}_{:s}\t{3.1f}\n"
    # get where in the lower triangle of the matrix (it is symmeric)
    # the value is greater than the persistence cut-off
    where_gt_perco = np.argwhere(np.tril(percmat>perco))
    for i, j in where_gt_perco:
        segid1, resid1, resname1 = shortids[shortids.index(shortidxs[i])]
        segid2, resid2, resname2 = shortids[shortids.index(shortidxs[j])]
        outstr += outstrfmt.format(segid1, resid1, resname1, idxs[i][3], \
                                   segid2, resid2, resname2, idxs[j][3], \
                                   percmat[i,j])
    # set the full matrix to None
    fullmatrix = None
    # compute the full matrix if requested
    if calc_fullmatrix_func is not None:
        fullmatrix = calc_fullmatrix_func(identifiers = identifiers, \
                                          idxs = idxs, \
                                          percmat = percmat, \
                                          perco = perco)
    # return output string and fullmatrix
    return (outstr, fullmatrix)



############################ HYDROGEN BONDS ###########################

def do_hbonds(sel1, \
              sel2, \
              refuni, \
              uni, \
              distance, \
              angle, \
              perco, \
              dofullmatrix, \
              otherhbs, \
              perresidue):
    
    # import the hydrogen bonds analysis module
    from MDAnalysis.analysis.hydrogenbonds.hbond_analysis \
    import HydrogenBondAnalysis
    # set up the hydrogen bonds analysis
    hbonds = HydrogenBondAnalysis(\
                 universe = uni, \
                 between = [sel1, sel2], \
                 d_a_cutoff = distance, \
                 d_h_a_angle_cutoff = angle)
    # get default donors
    defaultdonors = hbonds.guess_donors("protein")
    # get default acceptors
    defaultacceptors = hbonds.guess_acceptors("protein")
    # initialize donors and acceptors
    hbonds.donors_sel = defaultdonors
    hbonds.acceptors_sel = defaultacceptors
    # other donors and/or acceptors have been provided
    if otherhbs:
        # get the other donors
        otherdonors = " or ".join(\
            ["name {:s}".format(a) for a in otherhbs["DONORS"]])
        # get the other acceptors
        otheracceptors = " or ".join(\
            ["name {:s}".format(a) for a in otherhbs["ACCEPTORS"]])
        # update the list of donors
        hbonds.donors_sel = \
            f"({defaultdonors}) or ({otherdonors})"
        # update the list of acceptors
        hbonds.acceptors_sel = \
            f"({defaultacceptors}) or ({otheracceptors})"
    # inform the user about the hydrogen bond analysis parameters
    logstr = "Will use {:s}: {:s}"
    log.info(logstr.format("donors", hbonds.donors_sel))
    log.info(logstr.format("acceptors", hbonds.acceptors_sel))
    # run the hydrogen bonds analysis
    log.info("Running hydrogen bonds analysis ...")
    hbonds.run()
    log.info("Done! Finalizing ...")
    # get the hydrogen bonds timeseries
    data = hbonds.hbonds
    # create identifiers for the uni Universe
    uniidentifiers = [(res.segid, res.resid, res.resname, "residue") \
                       for res in uni.residues]
    # create identifiers for the reference Universe (reference)
    identifiers = [(res.segid, res.resid, res.resname, "residue") \
                   for res in refuni.residues]
    # map the identifiers of the uni Universe to their corresponding
    # indexes in the matrix
    uniid2uniix = dict([(item, i) for i, item in enumerate(uniidentifiers)])
    # uni IDs to refuni uni IDs
    uniid2refuniid = dict(zip(uniidentifiers, identifiers))
    # utility function to get the identifier of a hydrogen bond
    getid = lambda uni, hb: frozenset(((uni.atoms[hb[0]].segid, \
                                        uni.atoms[hb[0]].resid, \
                                        uni.atoms[hb[0]].resname, \
                                        "residue"), \
                                       (uni.atoms[hb[1]].segid, \
                                        uni.atoms[hb[1]].resid, \
                                        uni.atoms[hb[1]].resname, \
                                        "residue")))

    # create the full matrix if requested
    fullmatrix = None if not dofullmatrix \
                 else np.zeros((len(identifiers),len(identifiers)))
    # set the empty output string
    outstr = ""
    if perresidue or dofullmatrix:
        # get the number of frames in the trajectory
        numframes = len(uni.trajectory)
        # set the output string format
        outstrfmt = "{:s}-{:d}{:s}:{:s}-{:d}{:s}\t\t{:3.2f}\n"
        # data are in the form:
        # data = [
        #   [
        #       <frame>,
        #       <donor index (0-based)>,
        #       <hydrogen index (0-based)>,
        #       <acceptor index (0-based)>,
        #       <distance>,
        #       <angle>
        #   ],
        #   ...
        # ]
        # identify each hydrogen bond by its donor and acceptor
        # (neglect hydrogen atom at index 2)
        hbset = set([(int(item[1]), int(item[3])) for item in data])
        # get a list of the unique hydrogen bonds identified (between
        # pairs of residues) and convert their indexes to IDs
        hbids = list(set([getid(uni, hb) for hb in hbset]))
        # initialize an empty counter (number of occurrences
        # for each hydrogen bond)
        outdata = collections.Counter({hbid : 0 for hbid in hbids})
        # group data by frame
        dataperframe = \
            [zip(*list(zip(*g))[1:4]) for _, g \
             in itertools.groupby(data, key = lambda x:x[0])]
        for frame in dataperframe:
            # update the counter for the occurences of each
            # hydrogen bond in this frame
            # convert the indexes of donors and acceptors into
            # integers, since in the original data they were floats
            # (data are homogeneous numpy arrays)
            outdata.update(set([getid(uni, (int(don), int(acc))) \
                                for don, hydro, acc in frame]))
        # for each hydrogen bond identified in the trajectory
        for hbid, hboccur in outdata.items():
            # get the persistence of the hydrogen bond
            hbpers = (float(hboccur)/float(numframes))*100
            # convert the identifier from a frozenset to a list
            # to get items by index (the order does not matter)
            hbidlist = list(hbid)
            # get info about the first residue
            res1 = hbidlist[0]
            res1resix = uniid2uniix[res1]
            res1segid, res1resid, res1resname, res1string = \
                uniid2refuniid[res1]
            # get info about the second residue (if present)
            res2 = res1 if len(hbid) == 1 else hbidlist[1]
            res2resix = uniid2uniix[res2]
            res2segid, res2resid, res2resname, res2string = \
                uniid2refuniid[res2]
            # if the persistence of the hydrogen bond is higher
            # than the persistence threshold
            if hbpers > perco:
                # fill the full matrix if requested (it is symmetric)
                if dofullmatrix:
                    fullmatrix[res1resix, res2resix] = hbpers
                    fullmatrix[res2resix, res1resix] = hbpers
                # format the output string if requested
                if perresidue:
                    outstr += outstrfmt.format(\
                                res1segid, res1resid, res1resname, \
                                res1segid, res1resid, res2resname, \
                                hbpers)
    
    # do not merge hydrogen bonds per residue
    if not perresidue:
        # count hydrogen bonds by type
        table = h.count_by_type()
        # get the hydrogen bonds identifiers
        hbids = [list(getid(uni, hb)) for hb in table]
        # set output string format
        outstrfmt = "{:s}-{:d}{:s}_{:s}:{:s}-{:d}{:s}_{:s}\t\t{:3.2f}\n"
        # for each hydrogen bonds identified
        for i, hbid in enumerate(hbids):
            # get the hydrogen bond persistence
            hbpers = table[i][-1]*100
            # get donor heavy atom and acceptor atom
            donor = table[i][4]
            acceptor = table[i][8]
            # consider only those hydrogen bonds whose persistence
            # is greater than the cut-off
            if hbpers > perco:
                # components of the identifier of the first residue
                res1segid, res1resid, res1resname, res1tag = \
                    identifiers[uniid2uniix[hbid[0]]]
                # components of the identifier of the second residue
                res2segid, res2resid, res2resname, res2tag = \
                    identifiers[uniid2uniix[hbid[1]]]
                # update the output string
                outstr += outstrfmt.format(\
                            res1segid, res1resid, res1resname, donor, \
                            res2segid, res2resid, res2resname, acceptor, \
                            hbpers)

    # return output string and full matrix
    return (outstr, fullmatrix)
