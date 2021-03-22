#!/usr/bin/env python
# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-

import os
import os.path
import numpy as np
import pytest
from numpy.testing import assert_almost_equal, assert_equal

from pyinteraph import filter_graph as fg
from pyinteraph import graph_analysis as ga


######################## MODULE-LEVEL FIXTURES ########################

@pytest.fixture(scope = "module")
def ref_dir(request):
    return os.path.join(request.fspath.dirname, 'data/single_chain')

@pytest.fixture(scope = "module")
def pdb_fname(ref_dir):
    return os.path.join(ref_dir, "sim.prot.A.pdb")

@pytest.fixture(scope = "module")
def matrices_fnames(ref_dir):
    return [ os.path.join(ref_dir, "hc-graph.dat"),
             os.path.join(ref_dir, "hc-graph-filtered.dat") ]


@pytest.fixture(scope = "module")
def matrices(matrices_fnames):
    return [np.loadtxt(fname) for fname in matrices_fnames]

@pytest.fixture(scope = "module")
def results_dir(ref_dir):
    return os.path.join(ref_dir)


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
        return fg.get_maxclustsizes(matrices = matrices,
                                    interval = interval)

    @pytest.fixture(scope = "class")
    def args(self, interval, maxfev, p0):
        return fg.perform_fitting(f = fg.sigmoid,
                                  xdata = interval,
                                  ydata = interval,
                                  maxfev = maxfev,
                                  p0 = p0)

    @pytest.fixture(scope = "class")
    def flex(self, x0, args, maxfev):
        flex, infodict, ier, mesg = fg.find_flex(x0 = x0,
                                                 args = args,
                                                 maxfev = maxfev,
                                                 func = fg.seconddevsigmoid)
        return flex

    @pytest.fixture(scope = "class")
    def matrix_filter(self):
        return 20.0

    @pytest.fixture(scope = "class")
    def weights(self):
        return None

    #---------------------------- Tests ------------------------------#

    @pytest.mark.parametrize("x, x0, k, m, n, expected",
                            [(1.0, 20.0, 2.0, 10.0, 20.0, 30.0)])
    def test_sigmoid(self, x, x0, k, m, n, expected):
        sigmoid = fg.sigmoid(x = x, x0 = x0, k = k, m = m, n = n)
        assert_almost_equal(actual = sigmoid,
                            desired = expected)

    @pytest.mark.parametrize("x, x0, k, l, m, expected",
                             [(1.0, 1.0, 1.0, 1.0, 1.0, 0.0)])
    def test_seconddevsigmoid(self, x, x0, k, l, m, expected):
        seconddev = fg.seconddevsigmoid(x = x, x0 = x0, k = k,
                                        l = l, m = m)
        assert_almost_equal(actual = seconddev,
                            desired = expected)

    def test_perform_plotting(self, interval, maxclustsizes,
                              args, flex, results_dir):
        out_plot = os.path.join(results_dir, "test_plot.pdf")
        return fg.perform_plotting(x = interval,
                                   y = maxclustsizes,
                                   lower = interval[0],
                                   upper = interval[-1],
                                   out_plot = out_plot,
                                   args = args,
                                   flex = flex,
                                   func_sigmoid = fg.sigmoid)

    def test_write_clusters(self, interval, maxclustsizes,
                            results_dir):
        out_clusters = os.path.join(results_dir, "test_clusters.dat")
        return fg.write_clusters(out_clusters = out_clusters,
                                 interval = interval,
                                 maxclustsizes = maxclustsizes)


    def test_write_dat(self, matrices, matrix_filter,
                       weights, results_dir):
        out_dat = os.path.join(results_dir, "test_outmatrix.dat")
        return fg.write_dat(matrices = matrices,
                            matrix_filter = matrix_filter,
                            out_dat = out_dat,
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
    def G(self, matrices_fnames, pdb_fname):
        matrix_fname = matrices_fnames[0]
        identifiers, G = ga.build_graph(fname = matrix_fname,
                                        pdb = pdb_fname)
        return G

    @pytest.fixture(scope = "class")
    def identifiers(self, matrices_fnames, pdb_fname):
        matrix_fname = matrices_fnames[1]
        identifiers, G = ga.build_graph(fname = matrix_fname,
                                        pdb = pdb_fname)
        return identifiers

    @pytest.fixture(scope = "class")
    def ccs(self, G):
        return ga.get_connected_components(G = G)

    @pytest.fixture(scope = "class")
    def hubs(self, G):
        return ga.get_hubs(G = G, \
                           min_k = 3,
                           sorting = "descending")

    @pytest.fixture(scope = "class")
    def source(self):
        return "A-56ILE"

    @pytest.fixture(scope = "class")
    def target(self):
        return "A-92LEU"

    @pytest.fixture(scope = "class")
    def paths(self, G, source, target):
        maxl = 3
        sort_paths_by = "cumulative_weight"
        return ga.get_paths(G = G,
                            source = source,
                            target = target,
                            maxl = maxl,
                            sort_paths_by = sort_paths_by)

    #---------------------------- Tests ------------------------------#

    def test_hubs(self, hubs):
        expected = [('A-69MET', 7), ('A-24LEU', 6), ('A-106LEU', 6), ('A-22ILE', 5), 
                    ('A-67PRO', 5), ('A-105ALA', 5), ('A-11ILE', 4), ('A-20ALA', 4),
                    ('A-35LEU', 4), ('A-56ILE', 4), ('A-71PHE', 4), ('A-78PRO', 4),
                    ('A-89LEU', 4), ('A-92LEU', 4), ('A-96TRP', 4), ('A-98PRO', 4),
                    ('A-102LEU', 4), ('A-14VAL', 3), ('A-26PHE', 3), ('A-32ILE', 3),
                    ('A-54VAL', 3), ('A-58VAL', 3), ('A-75VAL', 3), ('A-80ILE', 3),
                    ('A-87ILE', 3), ('A-91ILE', 3), ('A-109LEU', 3), ('A-127ALA', 3),
                    ('A-141ALA', 3)]
        assert_equal(hubs, expected)

    def test_ccs(self, ccs):
        expected = [{'A-2SER'}, {'A-3ARG'}, {'A-35LEU', 'A-4ALA', 'A-64PHE', 'A-20ALA',
                     'A-56ILE', 'A-105ALA', 'A-89LEU', 'A-63PRO', 'A-100ILE', 'A-71PHE',
                     'A-98PRO', 'A-7ILE', 'A-112LEU', 'A-18PRO', 'A-107ILE', 'A-66PRO',
                     'A-14VAL', 'A-87ILE', 'A-149ALA', 'A-26PHE', 'A-32ILE', 'A-106LEU',
                     'A-11ILE', 'A-109LEU', 'A-91ILE', 'A-24LEU', 'A-67PRO', 'A-80ILE',
                     'A-59PRO', 'A-58VAL', 'A-99VAL', 'A-39PHE', 'A-22ILE', 'A-54VAL',
                     'A-102LEU', 'A-19ALA', 'A-92LEU', 'A-69MET', 'A-96TRP', 'A-75VAL',
                     'A-8MET', 'A-52PHE'}, {'A-5LYS'}, {'A-6ARG'}, {'A-9LYS'}, {'A-10GLU'},
                     {'A-12GLN'}, {'A-13ALA'}, {'A-15LYS'}, {'A-16ASP'}, {'A-17ASP'},
                     {'A-21HIS'}, {'A-23THR'}, {'A-25GLU'}, {'A-27VAL'}, {'A-28SER'},
                     {'A-29GLU'}, {'A-30SER'}, {'A-31ASP'}, {'A-33HIS'}, {'A-34HIS'},
                     {'A-36LYS'}, {'A-37GLY'}, {'A-38THR'}, {'A-40LEU'}, {'A-41GLY'},
                     {'A-42PRO', 'A-113LEU'}, {'A-43PRO'}, {'A-44GLY'}, {'A-45THR'},
                     {'A-137PHE', 'A-141ALA', 'A-46PRO', 'A-142ALA'}, {'A-47TYR'},
                     {'A-48GLU'}, {'A-49GLY'}, {'A-50GLY'}, {'A-51LYS'}, {'A-53VAL'},
                     {'A-55ASP'}, {'A-57GLU'}, {'A-60MET'}, {'A-61GLU'}, {'A-62TYR'},
                     {'A-65LYS'}, {'A-68LYS'}, {'A-70GLN'}, {'A-72ASP'}, {'A-73THR'}, 
                     {'A-74LYS'}, {'A-76TYR'}, {'A-77HIS'}, {'A-131LEU', 'A-127ALA', 
                     'A-121PRO', 'A-118PRO', 'A-78PRO', 'A-116PRO', 'A-124ALA', 
                     'A-126VAL'}, {'A-79ASN'}, {'A-81SER'}, {'A-82SER'}, {'A-83VAL'}, 
                     {'A-84THR'}, {'A-85GLY'}, {'A-86ALA'}, {'A-88CYS'}, {'A-90ASP'}, 
                     {'A-93LYS'}, {'A-94ASN'}, {'A-95ALA'}, {'A-97SER'}, {'A-101THR'}, 
                     {'A-103LYS'}, {'A-104SER'}, {'A-108SER'}, {'A-110GLN'}, {'A-111ALA'}, 
                     {'A-114GLN'}, {'A-115SER'}, {'A-117GLU'}, {'A-119ASN'}, {'A-120ASP'},
                     {'A-122GLN'}, {'A-123ASP'}, {'A-125GLU'}, {'A-128GLN'}, {'A-129HIS'},
                     {'A-130TYR'}, {'A-132ARG'}, {'A-133ASP'}, {'A-134ARG'}, {'A-135GLU'},
                     {'A-136SER'}, {'A-138ASN'}, {'A-139LYS'}, {'A-140THR'}, {'A-143LEU', 'A-147LEU'},
                     {'A-144TRP'}, {'A-145THR'}, {'A-146ARG'}, {'A-148TYR'}, {'A-150SER'}]
        assert_equal(ccs, expected)

    def test_paths(self, paths):
        expected = [(['A-56ILE', 'A-69MET', 'A-92LEU'], 3, 97.4, 48.7),
                    (['A-56ILE', 'A-69MET', 'A-105ALA', 'A-92LEU'], 4, 93.4, 31.133333333333336),
                    (['A-56ILE', 'A-54VAL', 'A-69MET', 'A-92LEU'], 4, 32.5, 10.833333333333334), 
                    (['A-56ILE', 'A-102LEU', 'A-105ALA', 'A-92LEU'], 4, 4.7, 1.5666666666666667)]
        print(expected)
        print("-------")
        print(paths)
        assert_equal(paths, expected)

    def test_get_resnum(self, resstring):
        return ga.get_resnum(resstring = resstring)

    def test_write_connected_components(self, ccs):
        return ga.write_connected_components(ccs = ccs,
                                             outfile = None)

    def test_write_connected_components_pdb(self, identifiers, ccs, \
                                            pdb_fname, results_dir):
        components_pdb = os.path.join(results_dir, "test_ccs.pdb")
        return ga.write_connected_components_pdb(\
                    identifiers = identifiers,
                    ccs = ccs,
                    ref = pdb_fname,
                    components_pdb = components_pdb,
                    replace_bfac_func = ga.replace_bfac_column)

    def test_write_hubs(self, hubs):
        return ga.write_hubs(hubs = hubs,
                             outfile = None)

    def test_write_hubs_pdb(self, identifiers, hubs, \
                            pdb_fname, results_dir):
        hubs_pdb = os.path.join(results_dir, "test_hubs.pdb")
        return ga.write_hubs_pdb(
                identifiers = identifiers,
                hubs = hubs,
                ref = pdb_fname,
                hubs_pdb = hubs_pdb,
                replace_bfac_func = ga.replace_bfac_column)

    def test_write_paths(self, paths):
        return ga.write_paths(paths = paths,
                              outfile = None)

    def test_write_paths_matrices(self, identifiers,
                                  G, paths, results_dir):
        return ga.write_paths_matrices(identifiers = identifiers,
                                       G = G,
                                       paths = paths,
                                       fmt = "%.1f",
                                       where = results_dir)
