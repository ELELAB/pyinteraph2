import pytest
import numpy as np
import os
from libinteract import libinteract as li
from numpy.testing import *
import MDAnalysis as mda
import pkg_resources
import configparser as cp

@pytest.fixture
def potential_file():
    return pkg_resources.resource_filename('pyinteraph', "ff.S050.bin64")

@pytest.fixture
def kbp_atoms_file():
    return pkg_resources.resource_filename('pyinteraph', "kbp_atomlist")

@pytest.fixture
def masses_file():
    return os.path.join(pkg_resources.resource_filename('pyinteraph', 'ff_masses'), 'charmm27')

@pytest.fixture
def cg_file():
    return pkg_resources.resource_filename('pyinteraph', "charged_groups.ini")

@pytest.fixture
def hb_file():
    return pkg_resources.resource_filename('pyinteraph', "hydrogen_bonds.ini")

@pytest.fixture
def ref_dir(request):
    return os.path.join(request.fspath.dirname, '../examples/tutorial')

@pytest.fixture
def data_files(ref_dir):
    return { 
             'gro' : os.path.join(ref_dir, 'sim.prot.gro'),
             'xtc' : os.path.join(ref_dir, 'traj.xtc'),
             'pdb' : os.path.join(ref_dir, 'sim.prot.A.pdb')
           }

@pytest.fixture
def ref_sb_file(ref_dir):
    return '{0}/salt-bridges.dat'.format(ref_dir)

@pytest.fixture
def ref_hb_file(ref_dir):
    return '{0}/hydrogen-bonds.dat'.format(ref_dir)

@pytest.fixture
def ref_hc_file(ref_dir):
    return '{0}/hydrophobic-clusters.dat'.format(ref_dir)

@pytest.fixture
def ref_sb_graph_file(ref_dir):
    return '{0}/sb-graph.dat'.format(ref_dir)

@pytest.fixture
def ref_hb_graph_file(ref_dir):
    return '{0}/hb-graph.dat'.format(ref_dir)

@pytest.fixture
def ref_hc_graph_file(ref_dir):
    return '{0}/hc-graph.dat'.format(ref_dir)

@pytest.fixture
def ref_sb(ref_sb_file):
    with open(ref_sb_file) as fh:
        return fh.readlines()

@pytest.fixture
def ref_hb(ref_hb_file):
    with open(ref_hb_file) as fh:
        return fh.readlines()

@pytest.fixture
def ref_hc(ref_hc_file):
    with open(ref_hc_file) as fh:
        return fh.readlines()

@pytest.fixture
def ref_sb_graph(ref_sb_graph_file):
    return np.loadtxt(ref_sb_graph_file)

@pytest.fixture
def ref_hb_graph(ref_hb_graph_file):
    return np.loadtxt(ref_hb_graph_file)

@pytest.fixture
def ref_hc_graph(ref_hc_graph_file):
    return np.loadtxt(ref_hc_graph_file)

@pytest.fixture
def ref_potential_file(ref_dir):
    return "{0}/kb-potential.dat".format(ref_dir)

@pytest.fixture
def ref_potential_graph_file(ref_dir):
    return "{0}/kbp-graph.dat".format(ref_dir)

@pytest.fixture
def ref_potential(ref_potential_file):
    with open(ref_potential_file) as fh:
        return fh.readlines()

@pytest.fixture
def ref_potential_graph(ref_potential_graph_file):
    return np.loadtxt(ref_potential_graph_file)

@pytest.fixture
def sparse_list():
    return np.arange(0, 10)

@pytest.fixture
def sparse_obj(sparse_list):
    return li.Sparse(sparse_list)

@pytest.fixture
def sparse_bin():
    return ['a', 'b', 'c', 'd', 1]

@pytest.fixture
def kbp_atomlist(kbp_atoms_file):
    return li.parse_atomlist(kbp_atoms_file)

@pytest.fixture
def sc_residues_list():
    return ["ALA", "ARG", "ASN", "ASP",
            "CYS", "GLN", "GLU", "HIS",
            "ILE", "LEU", "LYS", "MET",
            "PHE", "PRO", "SER", "THR",
            "TRP", "TYR", "VAL"]

@pytest.fixture
def hc_residues_list():
    return ['ALA', 'VAL', 'LEU', 'ILE',
            'PHE', 'PRO', 'TRP', 'MET'] 

@pytest.fixture
def simulation(data_files):
    uni = mda.Universe(data_files['gro'], data_files['xtc'])        
    pdb = mda.Universe(data_files['pdb'])
    return {'pdb' : pdb,
            'uni' : uni}

@pytest.fixture
def charged_groups(cg_file):
    return li.parse_cgs_file(cg_file)

@pytest.fixture
def hb_don_acc(hb_file):
    return li.parse_hbs_file(hb_file)

class TestSparse:
    def test_Sparse_constructor(self, sparse_list, sparse_obj):
        data = np.array([   sparse_obj.r1,
                            sparse_obj.r2,
                            sparse_obj.p1_1,
                            sparse_obj.p1_2,
                            sparse_obj.p2_1,
                            sparse_obj.p2_2,
                            sparse_obj.cutoff**2,
                            1.0 / sparse_obj.step,
                            sparse_obj.total,
                            sparse_obj.num])
        assert_almost_equal(data, sparse_list, decimal=10)

    def test_add_bin(self, sparse_obj, sparse_bin):
        sparse_obj.add_bin(sparse_bin)

        key = ''.join(sparse_bin[0:4])
        val = sparse_bin[4]

        assert_equal(sparse_obj.bins[key], val)

    def test_num_bins(self, sparse_obj, sparse_bin):
        sparse_obj.add_bin(sparse_bin)

        assert_equal(len(sparse_obj.bins), 1)

def test_parse_sparse(potential_file):
    li.parse_sparse(potential_file)

def test_parse_atomlist(kbp_atoms_file):
    data = li.parse_atomlist(kbp_atoms_file)

    assert(len(data) == 20)
    assert(data['GLN'] == ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD', 'OE1', 'NE2'])

def test_do_potential(kbp_atomlist, sc_residues_list, potential_file, simulation, ref_potential, ref_potential_graph):
    str_out, mat_out = li.do_potential(kbp_atomlist,
                       sc_residues_list,
                       potential_file,
                       uni = simulation['uni'],
                       pdb = simulation['pdb'],
                       do_fullmatrix = True)

    assert_almost_equal(mat_out, ref_potential_graph, decimal=3)
    split_str = str_out.split("\n")[:-1]
    for i, s in enumerate(split_str):
        assert(s == ref_potential[i].strip())

def test_ff_masses(simulation, masses_file):

    sel = [ simulation['uni'].select_atoms("resid 10 and not backbone") ]

    li.assign_ff_masses(masses_file, sel)

def test_generate_cg_identifiers(simulation, charged_groups):
    li.generate_cg_identifiers(pdb = simulation['pdb'],
                               uni = simulation['uni'],
                               cgs = charged_groups)

def test_generate_sc_identifiers(simulation, hc_residues_list):
    li.generate_sc_identifiers(pdb = simulation['pdb'],
                               uni = simulation['uni'],
                               reslist = hc_residues_list)

def test_parse_cg_files(cg_file):
    data = li.parse_cgs_file(cg_file)

def test_do_interact_sb(simulation, charged_groups, ref_sb_graph, ref_sb):
    str_out, sb_mat_out = li.do_interact(li.generate_cg_identifiers,
                                        pdb = simulation['pdb'],
                                        uni = simulation['uni'],
                                        co = 4.5,
                                        perco = 0,
                                        ffmasses = 'charmm27',
                                        fullmatrixfunc = li.calc_cg_fullmatrix,
                                        mindist = True,
                                        mindist_mode = 'diff',
                                        cgs = charged_groups)

    assert_almost_equal(sb_mat_out, ref_sb_graph, decimal=1)
    split_str = str_out.split("\n")[:-1]
    for i, s in enumerate(split_str):
        assert(s == ref_sb[i].strip())

def test_do_interact_hc(simulation, hc_residues_list, ref_hc_graph, ref_hc):
    str_out, hc_mat_out = li.do_interact(li.generate_sc_identifiers,
                                     pdb = simulation['pdb'],
                                     uni = simulation['uni'],
                                     co = 5.0, 
                                     perco = 0.0,
                                     ffmasses = 'charmm27', 
                                     fullmatrixfunc = li.calc_sc_fullmatrix, 
                                     reslist = hc_residues_list,
                                     mindist = False)
    assert_almost_equal(hc_mat_out, ref_hc_graph, decimal=1)
    split_str = str_out.split("\n")[:-1]
    for i, s in enumerate(split_str):
        assert(s == ref_hc[i].strip())

def test_parse_hb_file(hb_file):
    li.parse_hbs_file(hb_file)

def test_do_interact_hb(simulation, hb_don_acc, ref_hb, ref_hb_graph):
    sel = 'protein'
    str_out, hb_mat_out = li.do_hbonds(sel1 = sel,
                                   sel2 = sel,
                                   pdb = simulation['pdb'],
                                   uni = simulation['uni'],
                                   distance = 3.5,
                                   angle = 120.0,
                                   perco = 0.0,
                                   perresidue = False,
                                   do_fullmatrix = True,
                                   other_hbs = hb_don_acc)

    assert_almost_equal(hb_mat_out, ref_hb_graph, decimal=1)
    sorted_ref_hb = sorted(ref_hb)
    split_str = sorted(str_out.split("\n")[:-1])
    for i, s in enumerate(split_str):
        assert(s == sorted_ref_hb[i].strip())

