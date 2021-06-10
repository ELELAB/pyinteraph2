#!/usr/bin/env python
# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-
#
# Copyright (C) 2020, Valentina Sora <sora.valentina1@gmail.com>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful, 
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.

"""
Protein Structure Network calculation
=======================================================================
!!! Before using this module, check the integrity of the PBC
!!! of your trajectory because it cannot handle broken PBCs.

This module contains a base class to compute Protein Structure Networks
from sets of protein structures as described in Kannan and 
Vishveshwara, JMB, (1999)292,441-464.
"""


# Standard library
import collections.abc as abc
import copy
import logging
import math
import os
import pkg_resources
import sys

# Third-party packages
import MDAnalysis as mda
import MDAnalysis.analysis.distances as dist
import networkx as nx
import numpy as np

# Set logger
log = logging.getLogger(__name__)



class AtomicContactsPSNBuilder(object):
    """Class implementing a builder for a Protein Structure
    Network (PSN) based on atomic contacts as described in
    Kannan and Vishveshwara, JMB, (1999)292,441-464.
    """
    
    # Selecion string for only heavy side chains atoms 
    # (hydrogens excluded) 
    SELECTION_STRING = "protein and not backbone and prop mass >= 3"

    # Name to be given to the group of atoms used for the calculation
    # of atomic contacts
    SELECTION_NAME = "sidechain"


    def __init__(self):
        """Set up a Protein Network Analysis on a trajectory.
        The analysis is performed either parsing the trajectory
        frame by frame and getting the corresponding PSN(s) at
        each step with :meth:'PSNAnalysis.iter_psns()' or
        parsing the whole trajecory (or a subset of frames from
        the trajectory) and storing all the resulting PSNs with 
        :meth:'PSNAnalysis.get_psns()'.
        """
        pass    


    ######################### PRIVATE METHODS #########################


    def _set_i_min(self,
                   i_min):
        """Check and return a single i_min value.
        """
        
        # If the i_min is neither an integer, a float or None, raise
        # an error
        if not isinstance(i_min, (int, float, type(None))):
            errmsg = "A single i_min must be int, float or None."
            raise TypeError(errmsg)
        
        # Set the i_min to -inf if it was None
        i_min = float(i_min) if i_min else -np.inf
        
        # Inform the user about the i_min
        log.info(f"i_min provided: {i_min}.")
        
        # Return the i_min
        return float(i_min)


    def _set_i_mins(self,
                    i_mins):
        """Check and return the i_mins.
        """
        
        # If no i_mins were provided
        if not i_mins:
            log.info("No i_min provided. -inf will be used.")
            # Return an iterable only containin -np.inf
            return [-np.inf]
        
        # If i_mins were neither None nor an iterable
        if not isinstance(i_mins, abc.Iterable):
            errmsg = \
                "i_mins must be None or an iterable of int or float."
            raise TypeError(errmsg)

        # Inform the user about the i_min
        log.info(f"i_min(s) set to {', '.join(i_mins)}.")          
        
        # Return a list of i_mins
        return [self._set_i_min(i_min) for i_min in i_mins]


    def _set_dist_cut(self,
                      dist_cut):
        """Check and return the distance cutoff.
        """
        
        # dist_cut must be a number
        if not isinstance(dist_cut, (int, float)):
            raise TypeError("dist_cut must be a int or a float.")
        
        # dist_cut must be strictly greater than 0
        if dist_cut <= 0.0:
            raise ValueError("dist_cut must be greater than zero.")
        
        # Inform the user about the distance cut-off
        log.info(f"Distance cutoff set to {dist_cut}.")
        
        # Always convert to float before returning  
        return float(dist_cut)


    def _set_prox_cut(self,
                      prox_cut):
        """Check and return the proximity cutoff.
        """
        
        # If no proximity cut-off was provided
        if not prox_cut:
            # Return 0, that means no proximity cutoff
            # will be applied
            prox_cut = 0
        
        # If the proximity cut-off provided was not an integer
        if not isinstance(prox_cut, int):
            raise TypeError("prox_cut must be an integer or None.")
        
        # If the proximity cut-off provided was lower than 0
        if prox_cut < 0:
            errmsg = "prox_cut must be equal to or greater than zero."
            raise ValueError(errmsg)
        
        # Inform the user about the proximity cut-off
        log.info(f"Proximity cutoff set to {prox_cut}.")
        
        # Always convert to float before returning   
        return float(prox_cut)


    def _set_norm_facts(self,
                        norm_facts):
        """Check and return the default normalization 
        factors or those provided.
        """

        # Check the data type
        if not isinstance(norm_facts, (dict, type(None))):
            TypeError("norm_facts must be None or a dictionary.")
        
        # Inform the user about the normalization factors used
        normfactstr = "\n".join(\
            [f"{rt}: {nf:.4f}" for rt, nf in norm_facts.items()])    
        log.info(f"Normalization factors:\n{normfactstr}.")
        
        # Return the normalization factors     
        return norm_facts


    def _set_p_min(self,
                   p_min):
        """Check and return the p_min.
        """

        # Check the data type
        if not isinstance(p_min, (int, float, type(None))):
            raise TypeError("p_min must be an int, float or None.")
        
        # If no p_min was passed
        if not p_min:
            # Persistence cut-off set to zero (= no cut-off)
            p_min = 0.0
        
        # Inform the user about the persistence cut-off
        log.info(f"Persistence cutoff set to {p_min}.")
        
        # Always convert to float before returning
        return float(p_min)


    def _get_i_const(self,
                     norm_facts):
        """Calculate the i_const for pairs of residue types given
        their normalization factors. Return a dictionary of
        dictionaries of pre-computed multiplying factors to be used 
        in the I_ij calculation. There is a different multiplying
        factor for each pair of residue types, calculated as follows:
        
        _i_const[res1][res2] = 1/sqrt(norm_fact_res1*norm_fact_res2)
        """
        
        # Get the i_consts for all pairs of residues
        i_const =  \
            {res1 : \
                {res2 : \
                    1/math.sqrt(norm_facts[res1]* \
                                norm_facts[res2]) \
                 for res2 in norm_facts.keys()} \
             for res1 in norm_facts.keys()}
        
        # Return the i_const
        return i_const   


    def _get_residues(self,
                      universe,
                      universe_ref,
                      norm_facts):
        """Get each residue in the system that has a
        normalization factor associated.
        """

        # If the two Universes have a different number of
        # residues
        if len(universe.residues) != len(universe_ref.residues):
            errstr = \
                "The topology and the reference Universe have " \
                "a different number of residues."
            raise ValueError(errstr)

        # Get which residues of the system must be kept in
        # building the acPSN; use sets to speed up the lookup
        restokeep = [res for res in universe.residues \
                     if res.resname in set(norm_facts.keys())]

        # Get which residues of the reference must be kept
        restokeep_ref = [res for res in universe_ref.residues \
                         if res.resname in set(norm_facts.keys())]   
        
        # Inform the user about which residues will be included
        # in the construction of the PSN. For each residue, state
        # which is the corresponding one in the reference.
        logstr = \
            "\n".join(\
                [f"{r1.segid}-{r1.resnum}{r1.resname}, " \
                 f"reference: {r2.segid}-{r2.resnum}{r2.resname}"
                 for r1, r2 in zip(restokeep, restokeep_ref)])
        log.info(f"Residues considered for the construction " \
                 f"of the PSN:\n{logstr}")
        
        # Return the residues of the system and of the reference
        return restokeep, restokeep_ref


    def _get_nb_atoms_per_res(self,
                              universe,
                              dist_cut,
                              selstring):
        """Get the neighboring atoms of each residue in the
        system as an UpdatingAtomGroup.
        """

        # Inform the user about the selection string
        log.info(f"Atom selection: {selstring}")
        
        # Select the atoms based on the selection string
        sel = universe.select_atoms(selstring)
        
        # Split the filtered atom selection by residue    
        splitted_byres = sel.atoms.split(level = "residue")
        
        # Generate a list of length equal to len(splitted_byres) 
        # with tuples containing each an AtomGroup instance with the
        # atoms of each residue and an UpdatingAtomGroup with the
        # neighoring atoms of each residue
        nb_atoms_per_res = []
        for atoms_resi in splitted_byres:
            nb_atoms = sel.select_atoms(f"around {dist_cut} group sg ",
                                        sg = atoms_resi,
                                        updating = True,
                                        periodic = False)
            
            # atoms_resi[0] is the first atom of the AtomGroup created
            # by split() and representing each residue, therefore
            # we can extract the residue information from it
            segid = atoms_resi[0].segment.segid
            resnum = atoms_resi[0].residue.resnum
            resname = atoms_resi[0].residue.resname
            atomnames = ", ".join([atom.name for atom in nb_atoms])
            log.debug(f"Neighboring atoms selected for residue " \
                      f"{segid}-{resnum}{resname}: {atomnames}")
            
            # Append the atoms belonging to the residue and the
            # neighboring atoms to the list
            nb_atoms_per_res.append((atoms_resi,nb_atoms))
        
        # Return the list
        return nb_atoms_per_res


    def _compute_psn(self,
                     psn_shape,
                     nb_atoms_per_res,
                     prox_cut,
                     dist_cut,
                     i_const):
        """Compute a PSN according to the method proposed by
        Kannan and Vishveshwara, JMB, (1999)292,441-464.
        """

    
        #--------------------- PSN initialization --------------------#


        # Initialize a PSN filled with negative infinites
        psn = np.ones(shape = psn_shape, \
                      dtype = np.float64) \
              *(-np.inf)


        #------------------------- Residue 'i' -----------------------#


        # For each tuple of (atoms of residue 'i', 
        # neighboring atoms or residue 'i')
        for atoms_resi, nb_atoms in nb_atoms_per_res:
            
            # Get information about the residue and
            # the segment (= chain)
            resindex_i = atoms_resi[0].residue.ix
            resname_i = atoms_resi[0].residue.resname
            resnum_i = atoms_resi[0].residue.resnum
            segindex_i = atoms_resi[0].segment.ix
            segid_i = atoms_resi[0].segment.segid
            
            # Group the neighboring atoms by residue
            nb_atoms_byres = nb_atoms.split(level = "residue")
            
            # Create an empty list to store all residues
            # within the proximity cutoff of residue 'i'
            # (will be excluded)
            proxres = []


            #----------------------- Residue 'i' ---------------------#


            # For each group of neighboring atoms (each group
            # belonging to a different residue 'j')
            for atoms_resj in nb_atoms_byres:
                
                # Get information about the residue and
                # the segment (= chain)
                resindex_j = atoms_resj[0].residue.ix
                resname_j = atoms_resj[0].residue.resname
                resnum_j = atoms_resj[0].residue.resnum
                segindex_j = atoms_resj[0].segment.ix
                segid_j = atoms_resj[0].segment.segid


                #------------------ Proximity cut-off ----------------#


                # Ignore residue 'j' if it is within the proximity 
                # cutoff with respect to residue 'i' and they belong 
                # to the same segment
                is_same_segment = (segindex_i == segindex_j)
                is_within_prox_cut = \
                    resindex_j >= (resindex_i-prox_cut) and \
                    resindex_j <= (resindex_i+prox_cut)
                
                # Alternative formulation
                # is_within_prox_cut = \
                #    not (resindex_j < (resindex_i-prox_cut) or \
                #         resindex_j > (resindex_i+prox_cut))
                
                if is_same_segment and is_within_prox_cut:
                    proxres.append(f"{segid_j}-{resnum_j}{resname_j}")
                    continue


                #------------------ I_ij calculation -----------------#


                # Do not compute twice the pairs for each couple
                # of residues 'i' and 'j', therefore compute only
                # if the symmetric cell is still empty
                if psn[resindex_j, resindex_i] == (-np.inf):                
                    resname_j = atoms_resj[0].residue.resname
                    d_array = dist.distance_array(atoms_resi.positions, \
                                                  atoms_resj.positions)
                    
                    # Compute the number of atom pairs between the two
                    # residues having a distance lower or equal than
                    # the distance cutoff
                    atom_pairs = np.sum(d_array <= dist_cut)
                    
                    # Calculate the I_ij
                    I_ij = (atom_pairs * \
                            i_const[resname_i][resname_j]) \
                           * 100
                    
                    # Update the PSN
                    psn[resindex_i,resindex_j] = I_ij

            # If there were some proximal residues excluded
            if len(proxres) > 0:
                # Output all excluded residues for debug purposes
                log.debug(f"Residues excluded from neighbors of " \
                          f"residue {segid_i}-{resnum_i}{resname_i} " \
                          f"because of the proximity cutoff: " \
                          f"{','.join(proxres)}")

        # Return a symmetrized matrix (for a residue pair i,j 
        # the I_ij was computed and stored only once)
        return np.maximum(psn, psn.transpose())


    def _filter_psn_by_imin(self,
                            psn,
                            i_min):
        """Filter a PSN by a single i_min. Only values greater than
        or equal to i_min are retained, while the others are zeroed.
        """

        # Works also when i_min is -inf, because -inf < -inf is
        # evaluated as False
        psn[psn < i_min] = 0.0
        
        # Return the filtered PSN
        return psn 


    def _frame2psn(self,
                   i_mins,
                   psn_shape,
                   dist_cut,
                   prox_cut,
                   nb_atoms_per_res,
                   i_const):   
        """Compute and return the PSN(s) computed
        for the current frame.    
        """
        
        # Compute the PSN for the current frame
        psn = self._compute_psn(psn_shape = psn_shape,
                                dist_cut = dist_cut,
                                prox_cut = prox_cut,
                                nb_atoms_per_res = nb_atoms_per_res,
                                i_const = i_const)
        
        # If there are more than one i_mins
        if len(i_mins) > 1:
            # Collect the filtered PSNs
            psns = [self._filter_psn_by_imin(psn, i_min) \
                    for i_min in i_mins]          
            # Return a 3D array containing a 2D PSN for each i_min
            return np.stack(psns, axis = -1)
        
        # If there is only one i_min
        else:
            # Return the 2D PSN corresponding to the i_min provided
            return self._filter_psn_by_imin(psn, i_mins)

    
    def _iter_psns(self,
                   universe,
                   psn_shape,
                   i_mins,
                   dist_cut,
                   prox_cut,
                   i_const):
        """Create a generator to parse one frame of the trajectory
        at a time. At each call, it will return the PSN(s) computed
        for that frame.  
        """
        
        # Get the neighboring atoms per residue
        nb_atoms_per_res = self._get_nb_atoms_per_res(\
                                universe = universe,
                                selstring = self.SELECTION_STRING,
                                dist_cut = dist_cut)
        # for each frame
        for fs in universe.trajectory:
            
            # Log the progress in analyzing frames
            sys.stdout.write(f"\rAnalyzing frame: {fs.frame}")
            sys.stdout.flush()
            
            # Yield the PSN (or PSNs in case of multiple i_mins)
            yield self._frame2psn(\
                    i_mins = i_mins,
                    psn_shape = psn_shape,
                    dist_cut = dist_cut,
                    prox_cut = prox_cut,
                    nb_atoms_per_res = nb_atoms_per_res,
                    i_const = i_const)



    ########################### PUBLIC API ###########################



    def get_average_psn(self,
                        universe,
                        universe_ref,
                        norm_facts,
                        i_min = None,
                        dist_cut = 4.5,
                        prox_cut = None,
                        p_min = None,
                        edge_weight = "strength"):
        """Compute the average PSN over a set of 2D PSNs.
        """

        # If an invalid value was provided for the edge weight,
        # raise an error
        if edge_weight not in ("strength", "persistence"):   
            raise ValueError("edge_weight must be either " \
                             "'strength' or 'persistence'")

        # Check and return the dist_cut, prox_cut, norm_facts,
        # i_min, p_min
        dist_cut = self._set_dist_cut(dist_cut = dist_cut)
        prox_cut = self._set_prox_cut(prox_cut = prox_cut)
        norm_facts = self._set_norm_facts(norm_facts = norm_facts)
        i_min = self._set_i_min(i_min = i_min) 
        p_min = self._set_p_min(p_min = p_min)     
        
        # Compute the i_const values
        i_const = self._get_i_const(norm_facts = norm_facts)
        
        # Get the residues in the system and in the reference
        # for which a normalization factor is available
        residues, residues_ref = \
            self._get_residues(universe = universe,
                               universe_ref = universe_ref,
                               norm_facts = norm_facts)

        # Set the shape of the PSN (num. residues x num. residues)
        psn_shape = (len(residues), len(residues))
        
        # Create the iterator for the PSNs
        psns = self._iter_psns(universe = universe,
                               psn_shape = psn_shape,
                               i_mins = [i_min],
                               dist_cut = dist_cut,
                               prox_cut = prox_cut,
                               i_const = i_const)  
        
        # Set the average PSN to be the first PSN retrieved
        avg_psn = next(psns)
        
        # Set the edges occurrencies to be the edges occurrencies
        # of the first PSN retrieved 
        edges_occur = (avg_psn > 0.0).astype(int)
        
        # Set a counter for storing the total number of psns
        tot_psns = 1
        
        # For each PSN...
        for i, psn in enumerate(psns):
            # Add the current I_ij values to the cumulative average
            # PSN
            avg_psn = np.add(avg_psn, psn)
            # Increment the edges occurrencies in the cumulative
            # average PSN
            edges_occur = np.add(edges_occur, (psn > 0.0).astype(int))
            # Increament the counter of the total PSNs
            tot_psns += 1
        
        # Get the edges frequencies
        edges_freq = edges_occur / tot_psns
        
        # If the user chose to weight the edges on their persistence
        if edge_weight == "persistence":
            # Filter out those edges not exceeding
            # the persistence cutoff
            edges_freq[edges_freq<=p_min] = 0.0
            # Return the PSN of edge frequencies    
            final_psn = edges_freq
        
        # If the user chose to weight the edges on their strength
        elif edge_weight == "strength":
            # Filter out those edges not exceeding
            # the persistence cutoff
            avg_psn = avg_psn * ((edges_freq>p_min).astype(int))
            # Divide the final average PSN for the total number
            # of frames
            final_psn = (avg_psn / tot_psns)

        # Create a dictionary mapping the index of each residue in
        # the PSN to the chain the residue belongs to, its residue
        # number and name in the reference system
        ix2id = \
            {ix : (r.segid, r.resnum, r.resname) 
             for ix, r in enumerate(residues_ref)}

        # Create a table of contacts found in the final PSN (nonzero
        # positions in the PSN)
        nonzero_indexes = \
            [(r1, r2) for r1, r2 in np.argwhere(final_psn)]
        
        nonzero_values = \
            final_psn[tuple(zip(*nonzero_indexes))].tolist()

        table = \
            [(*ix2id[r1], self.SELECTION_NAME, \
              *ix2id[r2], self.SELECTION_NAME, \
              val) \
             for (r1, r2), val in zip(nonzero_indexes, nonzero_values)]

        # Return the table of contacts and the final PSN 
        return table, final_psn
