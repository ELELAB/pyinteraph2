#!/usr/bin/python

#    PyInteraph, a software suite to analyze interactions and interaction network in structural ensembles.
#    Copyright (C) 2013 Matteo Tiberti <matteo.tiberti@gmail.com>, Gaetano Invernizzi, Yuval Inbar, 
#    Matteo Lambrughi, Gideon Schreiber,  Elena Papaleo <elena.papaleo@unimib.it> <elena.papaleo@bio.ku.dk>
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.

#   This program is distributed in the hope that it will be useful,
#   but WITHOUT ANY WARRANTY; without even the implied warranty of
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#   GNU General Public License for more details.

#   You should have received a copy of the GNU General Public License
#   along with this program.  If not, see <http://www.gnu.org/licenses/>.

import logging
import numpy as np
cimport numpy as np

cimport cython
cimport innerloops

class LoopDistances():
    def __init__(self, coords1, coords2, co, corrections):
        self.coords1 = coords1
        self.coords2 = coords2
        self.co = co
        self.corrections = corrections

    def run_potential_distances(self, nsets_p, set_size_p, nframes_p):
        cdef int nsets = nsets_p
        cdef int set_size = set_size_p
        cdef int nframes = nframes_p
        cdef np.ndarray[np.float64_t, ndim=2] coords = self.coords1
        cdef np.ndarray[np.float64_t, ndim=1] results = np.zeros((nsets*nframes*4), dtype=np.float64)

        innerloops.potential_distances(<double*> coords.data, nsets, set_size, nframes, <double*> results.data)

        return np.reshape(results, (nframes,nsets,4))

    def run_triangular_distmatrix(self, natoms_p):
        cdef int natoms = natoms_p
        cdef int nframes = self.coords1.shape[0]/natoms_p
        cdef np.ndarray[np.int_t,    ndim=1] results = np.zeros((natoms*natoms), dtype=np.int)
        cdef np.ndarray[np.float64_t, ndim=2] coords1 = self.coords1
        cdef np.ndarray[np.float64_t, ndim=1] corrections = self.corrections

        innerloops.triangular_distmatrix(<double*> coords1.data, natoms, nframes, self.co, <long*> results.data, <double*> corrections.data)

        return np.reshape(results, (natoms,natoms))       

    def run_square_mindist(self, p_set_sizes1, p_set_sizes2):
        cdef int nframes = self.coords1.shape[0]/np.sum(p_set_sizes1)
        cdef int nsets1 = len(p_set_sizes1)
        cdef int nsets2 = len(p_set_sizes2)

        cdef np.ndarray[np.float64_t, ndim=2] coords1 = self.coords1
        cdef np.ndarray[np.float64_t, ndim=2] coords2 = self.coords2
        cdef np.ndarray[np.int_t,     ndim=1] set_sizes1 = p_set_sizes1
        cdef np.ndarray[np.int_t,     ndim=1] set_sizes2 = p_set_sizes2
        cdef np.ndarray[np.int_t,     ndim=1] results = np.zeros((nsets1*nsets2), dtype=np.int)

        #print set_sizes1
        #print set_sizes2
        #print self.coords1.shape
        #print self.coords2.shape

        #print "setsizes1", p_set_sizes1
        #print np.sum(p_set_sizes2)
	
        innerloops.square_mindist(<double*> coords1.data, <double*> coords2.data, nframes, nsets1, nsets2, <long*> set_sizes1.data, <long*> set_sizes2.data, self.co, <long*> results.data)
	
        return np.reshape(results, (nsets1, nsets2))	
	
    def run_triangular_mindist(self, p_set_sizes):

        cdef int nframes = self.coords1.shape[0]/np.sum(p_set_sizes)
        cdef int natoms  = self.coords1.shape[1]
        cdef int nsets   = len(p_set_sizes)

        cdef np.ndarray[np.float64_t, ndim=2] coords = self.coords1
        cdef np.ndarray[np.int_t,     ndim=1] set_sizes = p_set_sizes
        cdef np.ndarray[np.int_t,     ndim=1] results = np.zeros((nsets*nsets), dtype=np.int)
	
        innerloops.triangular_mindist(<double*> coords.data, nframes, nsets, <long*> set_sizes.data, self.co, <long*> results.data)
	
        return np.reshape(results, (nsets, nsets))	

