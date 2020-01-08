#!/usr/bin/env python
# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-
#
# Copyright (C) 2018, Valentina Sora <sora.valentina1@gmail.com>
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

import os
import os.path

#import MDAnalysis as mda
import numpy as np

#import pyinteraph as pyin
import filter_graph as fg
import graph_analysis as ga

import pytest
from numpy.testing import assert_almost_equal


TEST_DIR = os.path.join(os.getcwd(), "tests")
MATRICES_DIR = os.path.join(TEST_DIR, "matrices")
MATRICES_FNAMES = \
    [os.path.join(MATRICES_DIR, mat) for mat in os.listdir(MATRICES_DIR)]

@pytest.fixture(\
    scope = "module", \
    params = MATRICES_FNAMES)
def matrix_fname(request):
    return request.param

@pytest.fixture(scope = "module")
def matrices_fnames():
    return MATRICES_FNAMES

@pytest.fixture(scope = "module")
def matrices():
    return [np.loadtxt(mat) for mat in MATRICES_FNAMES]


class TestFilterGraph(object):

    RANGE_INTERVAL_STEP = np.arange(1,11)
    RANGE_MAXFEV = np.arange(0,100000,10000)

    @pytest.fixture(\
        scope = "class", \
        params = RANGE_INTERVAL_STEP)
    def interval(self, request):
        return np.arange(0,100,request.param)


    @pytest.fixture(\
        scope = "class", \
        params = RANGE_MAXFEV)
    def maxfev(self, request):
        return request.param


    @pytest.fixture(\
        scope = "class", \
        params = [(20.0, 2.0, 10.0, 20.0)])
    def p0(self, request):
        return request.param


    @pytest.mark.parametrize("x, x0, k, m, n, expected", \
                            [(1.0, 20.0, 2.0, 10.0, 20.0, 30.0)])
    def test_sigmoid(self, x, x0, k, m, n, expected):
        sigmoid = \
            fg.sigmoid(x = x, x0 = x0, k = k, m = m, n = n)
        assert_almost_equal(actual = sigmoid, \
                            desired = expected)


    @pytest.mark.parametrize("x, x0, k, l, m, expected", \
                             [(1.0, 1.0, 1.0, 1.0, 1.0, 0.0)])
    def test_seconddevsigmoid(self, x, x0, k, l, m, expected):
        seconddev = \
            fg.seconddevsigmoid(x = x, x0 = x0, k = k, l = l, m = m)
        assert_almost_equal(actual = seconddev, \
                            desired = expected)


    def test_load_matrix(self, matrix_fname):       
        return fg.load_matrix(fname = matrix_fname)


    def test_process_matrices(self, matrices_fnames):
        return fg.process_matrices(fnames = matrices_fnames)


    def test_get_maxclustsizes(self, matrices, interval):
        return fg.get_maxclustsizes(matrices = matrices, \
                                    interval = interval)


    def test_perform_fitting(self, interval, maxfev, p0):
        return fg.perform_fitting(f = fg.sigmoid, \
                                  xdata = interval, \
                                  ydata = interval, \
                                  maxfev = maxfev, \
                                  p0 = p0)


    def test_find_flex(self, x0, args, maxfev):
        pass


    def perform_plotting(self):
        pass


    def write_clusters(self, out_clusters, x):
        pass


    def write_dat(self, matrices, matrix_filter, out_dat, weights):
        pass



class TestGraphAnalysis(object):

    @pytest.fixture(\
        scope = "class", \
        params = ["A-42ALA", "SYSTEM-53SP2"])
    def resstring(self, request):
        return request.param


    @pytest.fixture(\
        scope = "class", \
        params = )


    def test_resstring2resnum(self, resstring):
        return ga.resstring2resnum(x = resstring)


    def test_replace_bfac_column(self, pdb, vals, pdb_out):
        pass


    def test_build_graph(self, fname, pdb):
        pass


    def test_get_connected_components(self, G):
        pass


    def test_write_connected_components(self, ccs, outfile):
        pass


    def test_write_connected_components_pdb(self, \
                                            identifiers, \
                                            ccs, \
                                            top, \
                                            components_pdb, \
                                            conversion_func):

        pass


