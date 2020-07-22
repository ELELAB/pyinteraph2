#!/usr/bin/env python
# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-

import os
import os.path
import numpy as np
import pytest
from numpy.testing import assert_almost_equal

import filter_graph as fg
import graph_analysis as ga


TEST_DIR = os.path.join(os.getcwd(), "tests")
MATRICES_DIR = os.path.join(TEST_DIR, "matrices")
RESULTS_DIR = os.path.join(TEST_DIR, "results")
PDB_FNAME = os.path.join(TEST_DIR, "files/ref.pdb")
MATRICES_FNAMES = \
    [os.path.join(MATRICES_DIR, mat) for mat in os.listdir(MATRICES_DIR)]


######################## MODULE-LEVEL FIXTURES ########################

@pytest.fixture(\
    scope = "module", \
    params = MATRICES_FNAMES)
def matrix_fname(request):
    return request.param

@pytest.fixture(scope = "module")
def pdb_fname():
    return PDB_FNAME

@pytest.fixture(scope = "module")
def matrices_fnames():
    return MATRICES_FNAMES

@pytest.fixture(scope = "module")
def matrices():
    return [np.loadtxt(mat) for mat in MATRICES_FNAMES]

@pytest.fixture(scope = "module")
def results_dir():
    return RESULTS_DIR


######################### FILTER GRAPH TESTS ##########################

class TestFilterGraph(object):

    #--------------------------- Fixtures ----------------------------#

    @pytest.fixture(scope = "class")
    def maxfev(self):
        return 20000

    @pytest.fixture(scope = "class")
    def p0(self):
        return (20.0, 2.0, 20.0, 10.0)

    @pytest.fixture(scope = "class")
    def x0(self):
        return 20.0

    @pytest.fixture(scope = "class")
    def interval(self):
        return np.arange(0,110,10)

    @pytest.fixture(scope = "class")
    def matrices(self, matrices_fnames):
        return fg.process_matrices(fnames = matrices_fnames)

    @pytest.fixture(scope = "class")
    def maxclustsizes(self, matrices, interval):
        return fg.get_maxclustsizes(matrices = matrices, \
                                    interval = interval)

    @pytest.fixture(scope = "class")
    def args(self, interval, maxfev, p0):
        return fg.perform_fitting(f = fg.sigmoid, \
                                  xdata = interval, \
                                  ydata = interval, \
                                  maxfev = maxfev, \
                                  p0 = p0)

    @pytest.fixture(scope = "class")
    def flex(self, x0, args, maxfev):
        flex, infodict, ier, mesg = fg.find_flex(x0 = x0, \
                                                 args = args, \
                                                 maxfev = maxfev, \
                                                 func = fg.seconddevsigmoid)
        return flex

    @pytest.fixture(scope = "class")
    def matrix_filter(self):
        return 20.0

    @pytest.fixture(scope = "class")
    def weights(self):
        return None

    #---------------------------- Tests ------------------------------#

    @pytest.mark.parametrize("x, x0, k, m, n, expected", \
                            [(1.0, 20.0, 2.0, 10.0, 20.0, 30.0)])
    def test_sigmoid(self, x, x0, k, m, n, expected):
        sigmoid = fg.sigmoid(x = x, x0 = x0, k = k, m = m, n = n)
        assert_almost_equal(actual = sigmoid, \
                            desired = expected)

    @pytest.mark.parametrize("x, x0, k, l, m, expected", \
                             [(1.0, 1.0, 1.0, 1.0, 1.0, 0.0)])
    def test_seconddevsigmoid(self, x, x0, k, l, m, expected):
        seconddev = fg.seconddevsigmoid(x = x, x0 = x0, k = k, \
                                        l = l, m = m)
        assert_almost_equal(actual = seconddev, \
                            desired = expected)

    def test_perform_plotting(self, interval, maxclustsizes, \
                              args, flex, results_dir):
        out_plot = os.path.join(results_dir, "test_plot.pdf")
        return fg.perform_plotting(x = interval, \
                                   y = maxclustsizes, \
                                   lower = interval[0], \
                                   upper = interval[-1], \
                                   out_plot = out_plot, \
                                   args = args, \
                                   flex = flex, \
                                   func_sigmoid = fg.sigmoid)

    def test_write_clusters(self, interval, maxclustsizes, \
                            results_dir):
        out_clusters = os.path.join(results_dir, "test_clusters.dat")
        return fg.write_clusters(out_clusters = out_clusters, \
                                 interval = interval, \
                                 maxclustsizes = maxclustsizes)


    def test_write_dat(self, matrices, matrix_filter, \
                       weights, results_dir):
        out_dat = os.path.join(results_dir, "test_outmatrix.dat")
        return fg.write_dat(matrices = matrices, \
                            matrix_filter = matrix_filter, \
                            out_dat = out_dat, \
                            weights = weights)


######################## GRAPH ANALYSIS TESTS #########################

class TestGraphAnalysis(object):

    #--------------------------- Fixtures ----------------------------#

    @pytest.fixture(\
        scope = "class", \
        params = ["A-42ALA", "SYSTEM-53SP2"])
    def resstring(self, request):
        return request.param

    @pytest.fixture(scope = "class")
    def G(self, matrix_fname, pdb_fname):
        identifiers, G = ga.build_graph(fname = matrix_fname, \
                                        pdb = pdb_fname)
        return G

    @pytest.fixture(scope = "class")
    def identifiers(self, matrix_fname, pdb_fname):
        identifiers, G = ga.build_graph(fname = matrix_fname, \
                                        pdb = pdb_fname)
        return identifiers

    @pytest.fixture(scope = "class")
    def ccs(self, G):
        return ga.get_connected_components(G = G)

    @pytest.fixture(scope = "class")
    def hubs(self, G):
        return ga.get_hubs(G = G, \
                           min_k = 3, \
                           sorting = "ascending")

    @pytest.fixture(scope = "class")
    def source(self):
        return "A-111CYS"

    @pytest.fixture(scope = "class")
    def target(self):
        return "A-114SER"

    @pytest.fixture(scope = "class")
    def paths(self, G, source, target):
        maxl = 10
        sort_paths_by = "cumulative_weight"
        return ga.get_paths(G = G, \
                            source = source, \
                            target = target, \
                            maxl = maxl, \
                            sort_paths_by = sort_paths_by)

    #---------------------------- Tests ------------------------------#

    def test_get_resnum(self, resstring):
        return ga.get_resnum(resstring = resstring)

    def test_write_connected_components(self, ccs):
        return ga.write_connected_components(ccs = ccs, \
                                             outfile = None)

    def test_write_connected_components_pdb(self, identifiers, ccs, \
                                            pdb_fname, results_dir):
        components_pdb = os.path.join(results_dir, "test_ccs.pdb")
        return ga.write_connected_components_pdb(\
                    identifiers = identifiers, \
                    ccs = ccs, \
                    ref = pdb_fname, \
                    components_pdb = components_pdb, \
                    replace_bfac_func = ga.replace_bfac_column)

    def test_write_hubs(self, hubs):
        return ga.write_hubs(hubs = hubs, \
                             outfile = None)

    def test_write_hubs_pdb(self, identifiers, hubs, \
                            pdb_fname, results_dir):
        hubs_pdb = os.path.join(results_dir, "test_hubs.pdb")
        return ga.write_hubs_pdb(\
                identifiers = identifiers, \
                hubs = hubs, \
                ref = pdb_fname, \
                hubs_pdb = hubs_pdb, \
                replace_bfac_func = ga.replace_bfac_column)

    def test_write_paths(self, paths):
        return ga.write_paths(paths = paths, \
                              outfile = None)

    def test_write_paths_matrices(self, identifiers, \
                                  G, paths, results_dir):
        return ga.write_paths_matrices(identifiers = identifiers, \
                                       G = G, \
                                       paths = paths, \
                                       fmt = "%.1f", \
                                       where = results_dir)