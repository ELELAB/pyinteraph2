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


    def _check_i_min(self,
                     i_min):
        """Check and return a single i_min value.
        """
        
        # If the i_min is neither an integer or a float or None, raise
        # an error
        if not isinstance(i_min, (int, float, None)):
            errstr = "A single i_min must be int, float or None."
            raise TypeError(errstr)
        
        # Set the i_min to 0.0 if None was passed
        if i_min is None:
            i_min = 0.0
            log.info(\
                "No i_min provided. 0.0 will be used as default i_min.")

        # Otherwise, convert the i_min to a float
        else:
            i_min = float(i_min)
            log.info(f"i_min provided: {i_min}.")
        
        # Return the i_min
        return float(i_min)


    def _check_i_mins(self,
                      i_mins):
        """Check and return the i_mins.
        """
        
        # If no i_min(s) was provided
        if i_mins is None:
            log.info(\
                "No i_min provided. 0.0 will be used as default i_min.")
            # Return an iterable only containing 0.0
            return [0.0]
        
        # If i_min(s) was neither provided nor an iterable
        if not isinstance(i_mins, abc.Iterable):
            errstr = \
                "i_mins must be None or an iterable of int or float."
            raise TypeError(errstr)         
        
        # Check the i_min(s)
        i_mins = [self._check_i_min(i_min) for i_min in i_mins]

        # Inform the user about the i_min(s)
        log.info(f"i_min(s) set to {', '.join(i_mins)}.")

        # Return the i_min(s)
        return i_mins


    def _check_dist_cut(self,
                        dist_cut):
        """Check and return the distance cutoff.
        """
        
        # dist_cut must be a number
        if not isinstance(dist_cut, (int, float)):
            raise TypeError("dist_cut must be a int or a float.")
        
        # dist_cut must be strictly greater than 0
        if dist_cut <= 0.0:
            errstr = \
                f"dist_cut must be greater than zero, but " \
                f"{dist_cut} was passed."
            raise ValueError(errstr)
        
        # Inform the user about the distance cut-off
        log.info(f"Distance cutoff set to {dist_cut}.")
        
        # Always convert to float before returning  
        return float(dist_cut)


    def _check_prox_cut(self,
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
            errstr = \
                f"prox_cut must be non negative, but {prox_cut} " \
                f"was passed."
            raise ValueError(errstr)
        
        # Inform the user about the proximity cut-off
        log.info(f"Proximity cutoff set to {prox_cut}.")
        
        # Return the proximity cut-off  
        return prox_cut


    def _check_norm_facts(self,
                          norm_facts):
        """Check and return the default normalization 
        factors or those provided.
        """

        # Check the data type
        if not isinstance(norm_facts, dict):
            raise TypeError("norm_facts must be a dictionary.")

        # Check that all normalization factors are greater than zero
        for res, norm_fact in norm_facts.items():
            if norm_fact <= 0.0:
                errstr = \
                    f"All normalization factors must be strictly " \
                    f"greater than zero, while the normalization " \
                    f"factor for {res} is {norm_fact}."
                raise ValueError(errstr)
        
        # Inform the user about the normalization factors used
        normfactstr = "\n".join(\
            [f"{rt}: {nf:.4f}" for rt, nf in norm_facts.items()])    
        log.info(f"Normalization factors:\n{normfactstr}.")
        
        # Return the normalization factors     
        return norm_facts


    def _check_p_min(self,
                     p_min):
        """Check and return the p_min.
        """

        # Check the data type
        if not isinstance(p_min, (int, float, type(None))):
            raise TypeError("p_min must be an int, float or None.")
        
        # If no p_min was passed
        if p_min is None:
            # Persistence cut-off set to zero (= no cut-off)
            p_min = 0.0

        # If a p_min was passed
        else:
            # Check that it is non negative
            if p_min < 0.0:
                errstr = \
                    f"p_min must be non negative, but {p_min} " \
                    f"was passed."
                raise ValueError(errstr)
        
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
                    (1/math.sqrt(norm_facts[res1]* \
                                 norm_facts[res2]))*100 \
                 for res2 in norm_facts.keys()} \
             for res1 in norm_facts.keys()}
        
        # Return the i_const
        return i_const   


    def _get_residues(self,
                      universe,
                      universe_ref,
                      norm_facts,
                      permissive,
                      norm_fact_default):
        """Get each residue in the system that has a
        normalization factor associated.
        """

        # Generate a copy of the dictionary of normalization
        # factors to be modified
        norm_facts_copy = dict(norm_facts)

        # If the two Universes have a different number of
        # residues
        if len(universe.residues) != len(universe_ref.residues):
            errstr = \
                "The topology and the reference Universe have " \
                "a different number of residues."
            raise ValueError(errstr)

        # Get which residues of the system must be kept in
        # building the acPSN; use sets to speed up the lookup
        restokeep = []
        
        # For each residue in the topology
        for res in universe.residues:
            
            # If the residue does not have an associated
            # normalization factor. Here, we are checking
            # on the copy of the original dictionary since
            # at this point is identical to the original.
            if not res.resname in set(norm_facts_copy.keys()):

                # If we are running in non-permisive mode
                if not permissive:

                    # Raise an error
                    errstr = \
                        f"Residue {res.resname} does not have an " \
                        f"associated normalization factor."
                    raise ValueError(errstr)
                
                # If we are running in permisive mode
                else:

                    # Add the residue name to the dictionary of
                    # normalization factors with the default
                    # normalization factor associated. Warn the
                    # user.
                    norm_facts_copy[res.resname] = norm_fact_default
                    warnstr = \
                        f"Residue {res.resname} has no associated " \
                        f"normalization factor. Since you are " \
                        f"running in permissive mode, the default " \
                        f"normalization factor ({norm_fact_default}) " \
                        f"will be assigned to it."
                    log.warnstr(warnstr)

            restokeep.append(res)

        # Get which residues of the reference must be kept
        restokeep_ref = []

        # For each residue in the reference
        for res in universe_ref.residues:

            # If the residue does not have an associated
            # normalization factor. Here, we are checking on
            # the copy of the original dictionary so that we
            # do not add twice residues which got assigned the
            # default normalization factor when parsing the
            # topology (it would have been overwritten but it
            # is still a waste of time).
            if not res.resname in set(norm_facts_copy.keys()):
                
                # If we are running in non-permisive mode
                if not permissive:

                    # Raise an error
                    errstr = \
                        f"Residue {res.resname} does not have an " \
                        f"associated normalization factor."
                    raise ValueError(errstr)

                # If we are running in permisive mode
                else:

                    # Add the residue name to the dictionary of
                    # normalization factors with the default
                    # normalization factor associated. Warn the
                    # user.
                    norm_facts_copy[res.resname] = norm_fact_default
                    warnstr = \
                        f"Residue {res.resname} has no associated " \
                        f"normalization factor. Since you are " \
                        f"running in permissive mode, the default " \
                        f"normalization factor ({norm_fact_default}) " \
                        f"will be assigned to it."
                    log.warnstr(warnstr)

            restokeep_ref.append(res)
        
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
        return restokeep, restokeep_ref, norm_facts_copy


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


            #----------------------- Residue 'j' ---------------------#


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


                # If residue 'i' and residue 'j' belong to the
                # same segment
                if segindex_i == segindex_j:
                    
                    # Check whether they are within the proximity
                    # cut-off
                    is_within_prox_cut = \
                        resindex_i - prox_cut <= resindex_j \
                        <= resindex_i + prox_cut

                    # If they are, append residue 'j' to the list
                    # of residues proximal to residue 'i' and
                    # continue
                    if is_within_prox_cut:
                        proxres.append(\
                            f"{segid_j}-{resnum_j}{resname_j}")
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
                    I_ij = atom_pairs * \
                           i_const[resname_i][resname_j]
                    
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
                        permissive,
                        norm_fact_default,
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
            errstr = \
                f"edge_weight must be either 'strength' or " \
                f"'persistence', but {edge_weight} was passed."
            raise ValueError(errstr)

        # Check and return the dist_cut, prox_cut, norm_facts,
        # i_min, p_min
        dist_cut = self._check_dist_cut(dist_cut = dist_cut)
        prox_cut = self._check_prox_cut(prox_cut = prox_cut)
        norm_facts = self._check_norm_facts(norm_facts = norm_facts)
        i_min = self._check_i_min(i_min = i_min) 
        p_min = self._check_p_min(p_min = p_min)

        # Get the residues in the system and in the reference
        # for which a normalization factor is available
        residues, residues_ref, norm_facts_copy = \
            self._get_residues(universe = universe,
                               universe_ref = universe_ref,
                               norm_facts = norm_facts,
                               permissive = permissive,
                               norm_fact_default = norm_fact_default)  
        
        # Compute the i_const values (use the updated copy of the
        # normalization factors' dictionary)
        i_const = self._get_i_const(norm_facts = norm_facts_copy)

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
        
        # Get the edges persistence
        edges_pers = (edges_occur / tot_psns) * 100
        
        # If the user chose to weight the edges on their persistence
        if edge_weight == "persistence":
            # Filter out those edges not exceeding
            # the persistence cutoff
            edges_pers[edges_pers<=p_min] = 0.0
            # Return the PSN of edges persistence    
            final_psn = edges_pers
        
        # If the user chose to weight the edges on their strength
        elif edge_weight == "strength":
            # Filter out those edges not exceeding
            # the persistence cutoff
            avg_psn = avg_psn * ((edges_pers>p_min).astype(int))
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
