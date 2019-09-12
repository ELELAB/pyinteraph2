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

major, minor, patch  = sys.version_info
if major < 2 or (major == 2 and minor < 7):
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


class LoopBreak(Exception):
    pass

class Sparse:
    def __repr__(self):
        fmt_repr = \
            "<Sparse r1={:d} ({:d},{:d}), r2={:d} ({:d},{:d}), {:d} bins>"
        
        return fmt_repr.format(self.r1, self.p1_1, self.p1_2, self.r2, \
                               self.p2_1, self.p2_2, self.binN())
    
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

    def addBin(self, bin):
        self.bins[''.join(bin[0:4])] = bin[4]

    def addBins(self, bins):
        for i in bins:
            self.bins[str(bins[0:4])] = bins[4]
    
    def binN(self):
        return len(self.bins)

kbp_reslist=  ["ALA","ARG","ASN","ASP","CYS","GLN","GLU","GLY","HIS","ILE","LEU","LYS","MET","PHE","PRO","SER","THR","TRP","TYR","VAL"]

def parse_sparse(ff):
    header_fmt  = '400i'
    header_struct = struct.Struct(header_fmt)
    header_size = struct.calcsize(header_fmt)
    sparse_fmt  = '=iiiiiidddidxxxx'
    sparse_size = struct.calcsize(sparse_fmt)
    sparse_struct = struct.Struct(sparse_fmt)
    bin_fmt     = '4cf'
    bin_size    = struct.calcsize(bin_fmt)
    bin_struct = struct.Struct(bin_fmt)
    pointer = 0
    sparses = []

    fh   = open(ff)
    data = fh.read()
    fh.close()

    isparse  = header_struct.unpack(data[pointer : pointer+header_size])
    pointer += header_size
    log.info("found %d residue-residue interaction definitions." % (len([i for i in isparse if i >0])))

    for i in isparse:
        sparses.append([])
        if i == 0:
            continue
        for j in range(i):
            this_sparse = Sparse(sparse_struct.unpack(data[pointer : pointer+sparse_size]))
            pointer += sparse_size
            for k in range(this_sparse.num): # for every bin....
                this_sparse.addBin(bin_struct.unpack(data[pointer : pointer+bin_size]))
                pointer += bin_size
            sparses[-1].append(this_sparse)
            
    assert pointer == len(data), "Error: could not completely parse the file (%d bytes read, %d expected)"%(pointer,len(data))

    sparses_dict = {}
    for i in range(len(kbp_reslist)):
        sparses_dict[kbp_reslist[i]] = {}
        for j in range(i):            
            sparses_dict[kbp_reslist[i]][kbp_reslist[j]] = {}
            
    for s in sparses:
        if s:
            sparses_dict[ kbp_reslist[ s[0].r1 ] ][ kbp_reslist[ s[0].r2 ] ] = s[0]
    print "Done!"
    return sparses_dict

def parse_atomlist(fname):
    try:
        fh = open(fname)
    except:
        log.error("could not open file %s. Exiting..."%fname)
        exit(1)
    data = {}
    for line in fh:
        tmp = line.strip().split(":")
        data[tmp[0].strip()] = [i.strip() for i in tmp[1].split(",")]
    fh.close()
    return data    
    
def dopotential(kbp_atomlist, residues_list, potential_file, seq_dist_co = 0, grof = None, xtcf = None, pdbf = None, uni = None, pdb = None, dofullmatrix = True, kbT=1.0):

    residues_list = ["ALA","ARG","ASN","ASP","CYS","GLN","GLU","HIS", "ILE","LEU","LYS","MET","PHE","PRO","SER","THR","TRP","TYR","VAL"] # Residues for which the potential is defined: all except G 

    log.info("Loading potential definition . . .")
    sparses = parse_sparse(potential_file)
    log.info("Loading input files...")

    if not pdb or not uni:
        if not pdbf or not grof or not xtcf:
            raise ValueError
        pdb,uni = loadsys(pdbf,grof,xtcf)

    ok_residues = []
    discarded_residues = set()
    residue_pairs = []
    atom_selections = []
    ordered_sparses = []
    numframes = len(uni.trajectory)

    for i in range(len(uni.residues)):
        if uni.residues[i].name in residues_list:
            ok_residues.append(i)
        else:
            discarded_residues.add(uni.residues[i])
            continue
        for j in ok_residues[:-1]:
            ii = i 
            if not (abs(i-j) < seq_dist_co or uni.residues[ii].segment.name != uni.residues[j].segment.name): 
                if uni.residues[j].name < uni.residues[ii].name:
                    ii,j = j,ii
                this_sparse = sparses[uni.residues[ii].name][uni.residues[j].name]
                this_atoms = (kbp_atomlist[uni.residues[ii].name][this_sparse.p1_1],
                              kbp_atomlist[uni.residues[ii].name][this_sparse.p1_2],
                              kbp_atomlist[uni.residues[j].name][this_sparse.p2_1],
                              kbp_atomlist[uni.residues[j].name][this_sparse.p2_2])
                try:
                    selected_atoms = mda.core.AtomGroup.AtomGroup((uni.residues[ii].atoms[uni.residues[ii].atoms.names().index(this_atoms[0])], 
                                      uni.residues[ii].atoms[uni.residues[ii].atoms.names().index(this_atoms[1])],
                                      uni.residues[j].atoms[uni.residues[j].atoms.names().index(this_atoms[2])],
                                      uni.residues[j].atoms[uni.residues[j].atoms.names().index(this_atoms[3])]))
                except: 
                    log.warning("could not identify essential atoms for the analysis (%s%s, %s%s)" % ( uni.residues[ii].name, uni.residues[ii].id, uni.residues[j].name, uni.residues[j].id ))
                    continue
                residue_pairs.append((ii,j))
                atom_selections.append(selected_atoms)
                ordered_sparses.append(this_sparse)

    scores = np.zeros((len(residue_pairs)), dtype=float)

    a=0    
    
    coords = None
    
    for ts in uni.trajectory:
        tmp_coords = []
        sys.stdout.write( "now analyzing: frame %d / %d (%3.1f%%)\r" % (a,numframes,float(a)/float(numframes)*100.0) )
        sys.stdout.flush()
        a+=1
        for sel in atom_selections:    
    	    tmp_coords.append(sel.coordinates())
        coords = np.array(np.concatenate(tmp_coords),dtype=np.float64)

        inner_loop = LoopDistances(coords, coords, None)

        distances = inner_loop.run_potential_distances(len(atom_selections), 4, 1)

        scores += calc_potential(distances, ordered_sparses, pdb, uni, seq_dist_co, kbT=kbT)
    
    scores /= float(len(uni.trajectory))
    outstr = ""

    for i,s in enumerate(scores):
        if abs(s) > 0.000001:
            outstr += "%s-%s%s:%s-%s%s\t%.3f\n" % (pdb.residues[residue_pairs[i][0]].segment.name, pdb.residues[residue_pairs[i][0]].name, pdb.residues[residue_pairs[i][0]].id, pdb.residues[residue_pairs[i][1]].segment.name, pdb.residues[residue_pairs[i][1]].name, pdb.residues[residue_pairs[i][1]].id, s)
        
    dm = None
    
    if dofullmatrix:
        dm = np.zeros((len(pdb.residues), len(pdb.residues)))
        for i,k in enumerate(residue_pairs):
            dm[k[0],k[1]] = scores[i]
            dm[k[1],k[0]] = scores[i]           
    
    return (outstr, dm)

def calc_potential(distances, ordered_sparses, pdb, uni, distco, kbT=1.0):

    general_cutoff=5.0
    
    tot=0
    done=0
    this_dist = np.zeros((2,2),dtype=np.float64)
    scores=np.zeros((distances.shape[1]))
    for frame in distances:
        for pn,this_dist in enumerate(frame):
            if np.any(this_dist < general_cutoff): #or this_dist[1] < general_cutoff or this_dist[2] < general_cutoff or this_dist[3] < general_cutoff:
                searchstring = ''.join([ chr(int(d * ordered_sparses[pn].step + 1.5)) for d in this_dist ])
                try:
                    probs = - kbT*ordered_sparses[pn].bins[searchstring]
                except KeyError:
                    probs = 0.0
                    pass
                scores[pn] += probs
            else:
                scores[pn] += 0.0
    return scores/distances.shape[0]

def load_gmxff(jsonfile):
    try:
        _jsonfile = open(jsonfile,'r')
    except: 
	raise
    return json.load(_jsonfile)

def distmatrix(uni,idxs,chosenselections,co,mindist=False, mindist_mode=None, type1char='p',type2char='n'):
    numframes = uni.trajectory.numframes
    final_percmat = np.zeros((len(chosenselections),len(chosenselections)))
    log.info("Distance matrix will be %dx%d (%d elements)" % (len(idxs),len(idxs),len(idxs)**2))
    a=1
    distmats=[]

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
                raise

        Nsizes = np.array(Nsizes, dtype=np.int)
        Psizes = np.array(Psizes, dtype=np.int)

	if mindist_mode == "diff":
            sets = [(P,N)]
            sets_idxs = [(Pidxs,Nidxs)]
            sets_sizes = [(Psizes,Nsizes)]
        elif mindist_mode == "same":
            sets = [(P,P),(N,N)]
            sets_idxs = [(Pidxs,Pidxs),(Nidxs,Nidxs)]
            sets_sizes = [(Psizes,Psizes),(Nsizes,Nsizes)]
        elif mindist_mode == "both":
            sets = [(chosenselections, chosenselections)]
            sets_idxs = [(idxs, idxs)]
            sizes =  [len(s) for s in chosenselections]
            sets_sizes = [(sizes,sizes)]
                
        else: raise

        percmats = []
        coords = []
        for s in sets:
            coords.append([[],[]])

        for ts in uni.trajectory:
            sys.stdout.write( "Caching coordinates: frame %d / %d (%3.1f%%)\r" % (a,numframes,float(a)/float(numframes)*100.0) )
            sys.stdout.flush()
            a+=1
            for si,s in enumerate(sets):
                if s[0] == s[1]: # triangular case
                    log.info("Caching coordinates...")
                    for group in s[0]:
                        coords[si][0].append(group.coordinates())
                        coords[si][1].append(group.coordinates())
                else: # square case
                    log.info("Caching coordinates...")
                    for group in s[0]:
                        coords[si][0].append(group.coordinates())
                    for group in s[1]:
                        coords[si][1].append(group.coordinates())

        for si,s in enumerate(sets): # recover the final matrix
            if s[0] == s[1]:
                this_coords = np.array(np.concatenate(coords[si][0]),dtype=np.float64)

                inner_loop = LoopDistances(this_coords, this_coords, co)
                percmats.append(inner_loop.run_triangular_mindist(sets_sizes[si][0]))

            else:
                this_coords1 = np.array(np.concatenate(coords[si][0]),dtype=np.float64)
                this_coords2 = np.array(np.concatenate(coords[si][1]),dtype=np.float64)
                
                inner_loop = LoopDistances(this_coords1, this_coords2, co)

                percmats.append( inner_loop.run_square_mindist(sets_sizes[si][0], sets_sizes[si][1]))

        for si,s in enumerate(sets): # recover the final matrix
            Pidxs = sets_idxs[si][0]
            Nidxs = sets_idxs[si][1]
            if s[0] == s[1]: # triangular case
                for j in range(len(s[0])):
                    for k in range(0,j):
                        final_percmat[idxs.index(Pidxs[j]), idxs.index(Pidxs[k])] = percmats[si][j,k]
                        final_percmat[idxs.index(Pidxs[k]), idxs.index(Pidxs[j])] = percmats[si][j,k]
            else: # square case
                for j in range(len(s[0])):
                    for k in range(len(s[1])):
                        final_percmat[idxs.index(Pidxs[j]), idxs.index(Nidxs[k])] = percmats[si][j,k]
                        final_percmat[idxs.index(Nidxs[k]), idxs.index(Pidxs[j])] = percmats[si][j,k]
 
        final_percmat = np.array(final_percmat, dtype=np.float)/numframes*100.0
                     
    else:
        all_coms = []
        for ts in uni.trajectory:
            sys.stdout.write( "now analyzing: frame %d / %d (%3.1f%%)\r" % (a,numframes,float(a)/float(numframes)*100.0) )
            sys.stdout.flush()
            a+=1
            distmat = np.zeros((len(chosenselections),len(chosenselections)))
            coms = np.zeros([len(chosenselections),3])
            for j in range(len(chosenselections)):
                coms[j,:] = chosenselections[j].centerOfMass()
            all_coms.append(coms)

        all_coms = np.concatenate(all_coms)
        inner_loop = LoopDistances(all_coms, all_coms, co)
        percmat = inner_loop.run_triangular_distmatrix(coms.shape[0])
        distmats = []
        final_percmat = np.array(percmat, dtype=np.float)/numframes*100.0


    return (final_percmat,distmats)

def loadsys(pdb,gro,xtc):
    
    
    uni = mda.Universe(gro,xtc)
        
    pdb = mda.Universe(pdb)
    return pdb,uni

def assignffmasses(ffmasses,idxs,chosenselections):
    ffdata = load_gmxff(ffmasses)
    for i in range(len(idxs)):
        for j in chosenselections[i]:
            try:                
                j.mass = ffdata[1][j.residue.name][j.name]
            except:
                log.warning("atom type not recognized (resid %s, atom %s). Atomic mass will be guessed." % (j.residue.name, j.name))
                pass    

def generateCustomIdentifiers(pdb,uni,**kwargs):
    selstrings = kwargs["selections"]
    names = kwargs["names"]
    assert len(names) == len(selstrings)
    chosenselections = []
    identifiers = []
    idxs = []
    for i in range(len(selstrings)):
        try:
            chosenselections.append(uni.selectAtoms(selstrings[i]))
            identifiers.append( ( names[i],"","","") )
            idxs.append( ( names[i], names[i], "", "", "" ) )
            print "Selection \"%s\" found with %d atoms" % (names[i], len(chosenselections[-1]))
        except:
            print "Warning: could not select \"%s\". Selection will be skipped." % names[i]
            continue
    return identifiers,idxs,chosenselections

def generateCGIdentifiers(pdb,uni,**kwargs):
    cgs = kwargs['cgs']
    identifiers = []
    idxs = []
    chosenselections = []
    for res,dic in cgs.iteritems(): # preprocess CGs: divide wolves and lambs
        for cgname,cg in dic.iteritems():
            cgs[res][cgname] =  {
                True:   set(                  filter(lambda x: not x.startswith("!"),cg)),   # atoms that must exist (negative of negative)
                False:  set( [ j[1:] for j in filter(lambda x:     x.startswith("!"),cg)])   # atoms that must NOT exist (positive of negative)
                }
    for j in range(len(pdb.residues)):
        identifiers.append( ( pdb.residues[j].segment.name, pdb.residues[j].id, pdb.residues[j].name, "" )  )
    for k in range(len(uni.residues)):
        curnames    = uni.residues[k].names()
        setcurnames = set(curnames)
        try:
            for cgname,cg in cgs[uni.residues[k].name].iteritems():
                if cg[True].issubset(setcurnames) and not bool(cg[False] & setcurnames): # if lambs exist AND wolves do not exist, add them

                    idxs.append( ( pdb.residues[k].segment.name, pdb.residues[k].id, pdb.residues[k].name, cgname )  )
                    chosenselections.append( mda.core.AtomGroup.AtomGroup( [ uni.residues[k][atom] for atom in cg[True] ] ) )    
                    log.info("%s %s (%s)" % (chosenselections[-1][0].resid, chosenselections[-1][0].resname, ", ".join([ l.name for l in chosenselections[-1]  ])))
        except KeyError:
            log.warning("residue %s is not in the charge recognition set. Will be skipped." % uni.residues[k].name)
    return identifiers,idxs,chosenselections
 
      
def generateSCIdentifiers(pdb,uni,**kwargs):
    reslist = kwargs['reslist']
    log.info("Selecting residues: %s" % ", ".join(reslist))
    excluded_atoms=["CA","C","O","N","H","H1","H2","H3","O1","O2","OXT","OT1","OT2"]
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
    
     
def SCFullmatrix(identifiers, idxs, percmat, perco):
    fullmatrix = np.zeros((len(identifiers),len(identifiers)))
    for i in range(len(identifiers)):
        for j in range(0,i):
            if identifiers[i] in idxs and identifiers[j] in idxs:
                fullmatrix[i,j] = percmat[idxs.index(identifiers[i]), idxs.index(identifiers[j])]
                fullmatrix[j,i] = percmat[idxs.index(identifiers[i]), idxs.index(identifiers[j])]
    return fullmatrix

def CGFullmatrix(identifiers, idxs, percmat, perco):
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
    
def dointeract(identfunc, grof=None, xtcf=None ,pdbf=None, pdb=None, uni=None, co=5.0, perco=0.0, ffmasses=None, fullmatrix=None, mindist=False, mindist_mode=None, **identargs ):

    outstr = ""

    idxs = []
    chosenselections = []
    identifiers = []
    
    if not pdb or not uni:
        if not pdbf or not grof or not xtcf:
            raise ValueError
        pdb,uni = loadsys(pdbf,grof,xtcf)
    
    identifiers,idxs,chosenselections = identfunc(pdb,uni,**identargs)

    if ffmasses is not None:
        try:
            assignffmasses(ffmasses,idxs,chosenselections)
        except IOError as (errno, strerror):
            log.warning("force field file not found or not readable. Masses will be guessed.")
            pass
    else:
        log.info("No force field assigned: masses will be guessed.")

    percmat,distmats = distmatrix(uni, idxs,chosenselections, co, mindist=mindist, mindist_mode=mindist_mode)

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

def dohbonds(sel1, sel2, grof=None ,xtcf=None, pdbf=None, pdb=None, uni=None, update_selection1=True, update_selection2=True, filter_first=False, distance=3.0, angle=120, perco=0.0, perresidue=False, dofullmatrix=False, other_hbs=None):
    
    outstr=""
    
    from MDAnalysis.analysis import hbonds
    
    if not pdb or not uni:
        if not pdbf or not grof or not xtcf:
            raise ValueError
        pdb,uni = loadsys(pdbf,grof,xtcf)


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
    
    

