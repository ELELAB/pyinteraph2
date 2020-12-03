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
    return os.path.join(request.fspath.dirname, '../examples')

@pytest.fixture
def data_files(ref_dir):
    return { 
             'gro' : os.path.join(ref_dir, 'sim.prot.gro'),
             'xtc' : os.path.join(ref_dir, 'traj.xtc'),
             'pdb' : os.path.join(ref_dir, 'sim.prot.A.pdb')
           }

@pytest.fixture
def data_twochains_files(ref_dir):
    return {
             'gro' : os.path.join(ref_dir, 'sim.prot.twochains.gro'),
             'xtc' : os.path.join(ref_dir, 'traj.twochains.xtc'),
             'pdb' : os.path.join(ref_dir, 'sim.prot.twochains.pdb')
           }

# table file names
@pytest.fixture
def ref_sb_file(ref_dir):
    return '{0}/salt-bridges.csv'.format(ref_dir)

@pytest.fixture
def ref_hb_file(ref_dir):
    return '{0}/hydrogen-bonds.csv'.format(ref_dir)

@pytest.fixture
def ref_hc_file(ref_dir):
    return '{0}/hydrophobic-clusters.csv'.format(ref_dir)

# matrix file names
@pytest.fixture
def ref_sb_graph_file(ref_dir):
    return '{0}/sb-graph.dat'.format(ref_dir)

@pytest.fixture
def ref_hb_graph_file(ref_dir):
    return '{0}/hb-graph.dat'.format(ref_dir)

@pytest.fixture
def ref_hc_graph_file(ref_dir):
    return '{0}/hc-graph.dat'.format(ref_dir)

# two chain table file names
@pytest.fixture
def ref_sb_twochains_file(ref_dir):
    return '{0}/salt-bridges_twochains_all_chains.csv'.format(ref_dir)

@pytest.fixture
def ref_hb_twochains_file(ref_dir):
    return '{0}/hydrogen-bonds_twochains_all_chains.csv'.format(ref_dir)

@pytest.fixture
def ref_hc_twochains_file(ref_dir):
    return '{0}/hydrophobic-clusters_twochains_all_chains.csv'.format(ref_dir)

# intra chain table file names
@pytest.fixture
def ref_sb_chains_file(ref_dir):
    return {
            'intra_A' : '{0}/salt-bridges_twochains_intra_chain_A.csv'.format(ref_dir),
            'inter' : '{0}/salt-bridges_twochains_inter_chain_B-A.csv'.format(ref_dir)
           }

@pytest.fixture
def ref_hc_chains_file(ref_dir):
    return {
            'intra_A' : '{0}/hydrophobic-clusters_twochains_intra_chain_A.csv'.format(ref_dir),
            'inter' : '{0}/hydrophobic-clusters_twochains_inter_chain_A-B.csv'.format(ref_dir)
           }

@pytest.fixture
def ref_hb_chains_file(ref_dir):
    return {
            'intra_A' : '{0}/hydrogen-bonds_twochains_intra_chain_A.csv'.format(ref_dir),
            'intra_B' : '{0}/hydrogen-bonds_twochains_intra_chain_B.csv'.format(ref_dir),
            'inter' : '{0}/hydrogen-bonds_twochains_inter_chain_A-B.csv'.format(ref_dir)
           }

# two chain matrix file names
@pytest.fixture
def ref_sb_graph_twochains_file(ref_dir):
    return '{0}/sb-graph_twochains_all_chains.dat'.format(ref_dir)

@pytest.fixture
def ref_hb_graph_twochains_file(ref_dir):
    return '{0}/hb-graph_twochains_all_chains.dat'.format(ref_dir)

@pytest.fixture
def ref_hc_graph_twochains_file(ref_dir):
    return '{0}/hc-graph_twochains_all_chains.dat'.format(ref_dir)

#inter chain matrix file names
@pytest.fixture
def ref_sb_graph_chains_file(ref_dir):
    return {
            'intra_A' : '{0}/sb-graph_twochains_intra_chain_A.dat'.format(ref_dir),
            'inter' : '{0}/sb-graph_twochains_inter_chain_B-A.dat'.format(ref_dir)
           }

@pytest.fixture
def ref_hc_graph_chains_file(ref_dir):
    return {
            'intra_A' : '{0}/hc-graph_twochains_intra_chain_A.dat'.format(ref_dir),
            'inter' : '{0}/hc-graph_twochains_inter_chain_A-B.dat'.format(ref_dir)
           }

@pytest.fixture
def ref_hb_graph_chains_file(ref_dir):
    return {
            'intra_A' : '{0}/hb-graph_twochains_intra_chain_A.dat'.format(ref_dir),
            'intra_B' : '{0}/hb-graph_twochains_intra_chain_B.dat'.format(ref_dir),
            'inter' : '{0}/hb-graph_twochains_inter_chain_A-B.dat'.format(ref_dir)
           }

# tables files
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

# inter chain table files
@pytest.fixture
def ref_sb_chains(ref_sb_chains_file):
    with open(ref_sb_chains_file['intra_A']) as A, \
         open(ref_sb_chains_file['inter']) as I:
        return {
                'intra_A' : A.readlines(),
                'inter' : I.readlines()
               }

@pytest.fixture
def ref_hc_chains(ref_hc_chains_file):
    with open(ref_hc_chains_file['intra_A']) as A, \
         open(ref_hc_chains_file['inter']) as I:
        return {
                'intra_A' : A.readlines(),
                'inter' : I.readlines()
               }

@pytest.fixture
def ref_hb_chains(ref_hb_chains_file):
    with open(ref_hb_chains_file['intra_A']) as A, \
         open(ref_hb_chains_file['intra_B']) as B, \
         open(ref_hb_chains_file['inter']) as I:
        return {
                'intra_A' : A.readlines(),
                'intra_B' : B.readlines(),
                'inter' : I.readlines()
               }

# inter chain matrix files
@pytest.fixture
def ref_sb_graph_chains(ref_sb_graph_chains_file):
    return {
            'intra_A' : np.loadtxt(ref_sb_graph_chains_file['intra_A']),
            'inter' : np.loadtxt(ref_sb_graph_chains_file['inter'])
            }

@pytest.fixture
def ref_hc_graph_chains(ref_hc_graph_chains_file):
    return {
            'intra_A' : np.loadtxt(ref_hc_graph_chains_file['intra_A']),
            'inter' : np.loadtxt(ref_hc_graph_chains_file['inter'])
            }

@pytest.fixture
def ref_hb_graph_chains(ref_hb_graph_chains_file):
    return {
            'intra_A' : np.loadtxt(ref_hb_graph_chains_file['intra_A']),
            'intra_B' : np.loadtxt(ref_hb_graph_chains_file['intra_B']),
            'inter' : np.loadtxt(ref_hb_graph_chains_file['inter'])
            }

# matrix files
@pytest.fixture
def ref_sb_graph(ref_sb_graph_file):
    return np.loadtxt(ref_sb_graph_file)

@pytest.fixture
def ref_hb_graph(ref_hb_graph_file):
    return np.loadtxt(ref_hb_graph_file)

@pytest.fixture
def ref_hc_graph(ref_hc_graph_file):
    return np.loadtxt(ref_hc_graph_file)

# two chains files
@pytest.fixture
def ref_sb_twochains(ref_sb_twochains_file):
    with open(ref_sb_twochains_file) as fh:
        return fh.readlines()

@pytest.fixture
def ref_hb_twochains(ref_hb_twochains_file):
    with open(ref_hb_twochains_file) as fh:
        return fh.readlines()

@pytest.fixture
def ref_hc_twochains(ref_hc_twochains_file):
    with open(ref_hc_twochains_file) as fh:
        return fh.readlines()

# two chain matrix files
@pytest.fixture
def ref_sb_graph_twochains(ref_sb_graph_twochains_file):
    return np.loadtxt(ref_sb_graph_twochains_file)

@pytest.fixture
def ref_hb_graph_twochains(ref_hb_graph_twochains_file):
    return np.loadtxt(ref_hb_graph_twochains_file)

@pytest.fixture
def ref_hc_graph_twochains(ref_hc_graph_twochains_file):
    return np.loadtxt(ref_hc_graph_twochains_file)

# potential files
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
def simulation_twochains(data_twochains_files):
    uni = mda.Universe(data_twochains_files['gro'], data_twochains_files['xtc'])
    pdb = mda.Universe(data_twochains_files['pdb'])
    return {'pdb' : pdb,
            'uni' : uni}

@pytest.fixture
def charged_groups(cg_file):
    return li.parse_cgs_file(cg_file)

@pytest.fixture
def hb_don_acc(hb_file):
    return li.parse_hbs_file(hb_file)

# do_interact functions
@pytest.fixture
def do_interact_sb(simulation_twochains, charged_groups):
    tab_out, mat_out = li.do_interact(li.generate_cg_identifiers,
                                      pdb = simulation_twochains['pdb'],
                                      uni = simulation_twochains['uni'],
                                      co = 4.5,
                                      perco = 0,
                                      ffmasses = 'charmm27',
                                      fullmatrixfunc = li.calc_cg_fullmatrix,
                                      mindist = True,
                                      mindist_mode = 'diff',
                                      cgs = charged_groups)

    return {'table': tab_out, 'matrix': mat_out}

@pytest.fixture
def do_interact_hc(simulation_twochains, hc_residues_list):
    tab_out, mat_out  = li.do_interact(li.generate_sc_identifiers,
                                       pdb = simulation_twochains['pdb'],
                                       uni = simulation_twochains['uni'],
                                       co = 5.0,
                                       perco = 0.0,
                                       ffmasses = 'charmm27',
                                       fullmatrixfunc = li.calc_sc_fullmatrix,
                                       reslist = hc_residues_list,
                                       mindist = False)


    return {'table': tab_out, 'matrix': mat_out}

@pytest.fixture
def do_interact_hb(simulation_twochains, hb_don_acc):
    sel = 'protein'
    tab_out, mat_out = li.do_hbonds(sel1 = sel,
                                    sel2 = sel,
                                    pdb = simulation_twochains['pdb'],
                                    uni = simulation_twochains['uni'],
                                    distance = 3.5,
                                    angle = 120.0,
                                    perco = 0.0,
                                    perresidue = False,
                                    do_fullmatrix = True,
                                    other_hbs = hb_don_acc)
    return {'table': tab_out, 'matrix': mat_out}

# table lists
@pytest.fixture
def create_table_dict_sb(do_interact_sb):
    return li.create_table_dict(do_interact_sb['table'])

@pytest.fixture
def create_table_dict_hc(do_interact_hc):
    return li.create_table_dict(do_interact_hc['table'])

@pytest.fixture
def create_table_dict_hb(do_interact_hb):
    return li.create_table_dict(do_interact_hb['table'], hb = True)

def split_dict(dictionary):
    keys = list(dictionary.keys())
    keys.remove("all")
    split_list = [dictionary[key] for key in keys]
    return split_list

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


# check do interact functions
def test_do_interact_sb(simulation, charged_groups, ref_sb_graph, ref_sb):
    table_out, sb_mat_out = li.do_interact(li.generate_cg_identifiers,
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
    for i, t in enumerate(table_out):
        assert(','.join(str(x) for x in t) == ref_sb[i].strip())

def test_do_interact_hc(simulation, hc_residues_list, ref_hc_graph, ref_hc):
    table_out, hc_mat_out = li.do_interact(li.generate_sc_identifiers,
                                           pdb = simulation['pdb'],
                                           uni = simulation['uni'],
                                           co = 5.0, 
                                           perco = 0.0,
                                           ffmasses = 'charmm27', 
                                           fullmatrixfunc = li.calc_sc_fullmatrix, 
                                           reslist = hc_residues_list,
                                           mindist = False)
    assert_almost_equal(hc_mat_out, ref_hc_graph, decimal=1)

    for i, t in enumerate(table_out):
        assert(','.join(str(x) for x in t) == ref_hc[i].strip())

def test_parse_hb_file(hb_file):
    li.parse_hbs_file(hb_file)

def test_do_interact_hb(simulation, hb_don_acc, ref_hb, ref_hb_graph):
    sel = 'protein'
    table_out, hb_mat_out = li.do_hbonds(sel1 = sel,
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
    for i, t in enumerate(table_out):
        assert(','.join(str(x) for x in t) == ref_hb[i].strip())

# check sb tables and matrix
def test_create_table_dict_sb(do_interact_sb, ref_sb_twochains, ref_sb_chains):
   table_dict = li.create_table_dict(do_interact_sb['table'])
   for i, t in enumerate(table_dict["all"]):
       assert(','.join(str(x) for x in t) == ref_sb_twochains[i].strip())
   for i, t in enumerate(table_dict['A']):
       assert(','.join(str(x) for x in t) == ref_sb_chains['intra_A'][i].strip())
   for i, t in enumerate(table_dict[('B', 'A')]):
       assert(','.join(str(x) for x in t) == ref_sb_chains['inter'][i].strip())

   first_table = np.sort(table_dict["all"], axis = 0)
   split_tables = np.sort(np.vstack(split_dict(table_dict)), axis = 0)
   assert(np.array_equal(first_table, split_tables) == True)
   
def test_create_matrix_dict_sb(do_interact_sb, create_table_dict_sb, simulation_twochains, ref_sb_graph_twochains, ref_sb_graph_chains):
    mat_dict = li.create_matrix_dict(do_interact_sb['matrix'],
                                     create_table_dict_sb,
                                     simulation_twochains['pdb'])
    assert_almost_equal(mat_dict["all"], ref_sb_graph_twochains, decimal=1)
    assert_almost_equal(mat_dict["A"], ref_sb_graph_chains['intra_A'], decimal=1)
    assert_almost_equal(mat_dict[("B", "A")], ref_sb_graph_chains['inter'], decimal=1)
    split_matrix = split_dict(mat_dict)
    # Ensure both matrices are of equal size and values
    assert_almost_equal(mat_dict["all"], sum(split_matrix), decimal=1)
    # Boundary test
    for i in range(len(split_matrix) - 1):
        common = np.logical_and(split_matrix[i] > 0, split_matrix[i+1] > 0)
        assert(common.sum() == 0)

# check hydrophobic tables and matrix
def test_create_table_dict_hc(do_interact_hc, ref_hc_twochains, ref_hc_chains):
   table_dict = li.create_table_dict(do_interact_hc['table'])
   for i, t in enumerate(table_dict["all"]):
       assert(','.join(str(x) for x in t) == ref_hc_twochains[i].strip())
   for i, t in enumerate(table_dict["A"]):
       assert(','.join(str(x) for x in t) == ref_hc_chains['intra_A'][i].strip())
   for i, t in enumerate(table_dict[("A", "B")]):
       assert(','.join(str(x) for x in t) == ref_hc_chains['inter'][i].strip())

   first_table = np.sort(table_dict["all"], axis = 0)
   split_tables = np.sort(np.vstack(split_dict(table_dict)), axis = 0)
   assert(np.array_equal(first_table, split_tables) == True)


def test_create_matrix_dict_hc(do_interact_hc, create_table_dict_hc, simulation_twochains, ref_hc_graph_twochains, ref_hc_graph_chains):
    mat_dict = li.create_matrix_dict(do_interact_hc['matrix'],
                                     create_table_dict_hc,
                                     simulation_twochains['pdb'])
    assert_almost_equal(mat_dict["all"], ref_hc_graph_twochains, decimal=1)
    assert_almost_equal(mat_dict["A"], ref_hc_graph_chains['intra_A'], decimal=1)
    assert_almost_equal(mat_dict[("A", "B")], ref_hc_graph_chains['inter'], decimal=1)
    split_matrix = split_dict(mat_dict)
    # Ensure both matrices are of equal size and values
    assert_almost_equal(mat_dict["all"], sum(split_matrix), decimal=1)
    # Boundary test
    for i in range(len(split_matrix) - 1):
        common = np.logical_and(split_matrix[i] > 0, split_matrix[i+1] > 0)
        assert(common.sum() == 0)

#check hydrogen bonds tables and matrix
def test_create_table_dict_hb(do_interact_hb, ref_hb_twochains, ref_hb_chains):
   table_dict = li.create_table_dict(do_interact_hb['table'], hb = True)
   for i, t in enumerate(table_dict["all"]):
       assert(','.join(str(x) for x in t) == ref_hb_twochains[i].strip())
   for i, t in enumerate(table_dict["A"]):
       assert(','.join(str(x) for x in t) == ref_hb_chains['intra_A'][i].strip())
   for i, t in enumerate(table_dict["B"]):
       assert(','.join(str(x) for x in t) == ref_hb_chains['intra_B'][i].strip())
   for i, t in enumerate(table_dict[("A", "B")]):
       assert(','.join(str(x) for x in t) == ref_hb_chains['inter'][i].strip())

   first_table = np.sort(table_dict["all"], axis = 0)
   split_tables = np.sort(np.vstack(split_dict(table_dict)), axis = 0)
   assert(np.array_equal(first_table, split_tables) == True)

def test_create_matrix_dict_hb(do_interact_hb, create_table_dict_hb, simulation_twochains, ref_hb_graph_twochains, ref_hb_graph_chains):
    mat_dict = li.create_matrix_dict(do_interact_hb['matrix'],
                                     create_table_dict_hb,
                                     simulation_twochains['pdb'],
                                     hb = True)
    assert_almost_equal(mat_dict["all"], ref_hb_graph_twochains, decimal=1)
    assert_almost_equal(mat_dict["A"], ref_hb_graph_chains['intra_A'], decimal=1)
    assert_almost_equal(mat_dict["B"], ref_hb_graph_chains['intra_B'], decimal=1)
    assert_almost_equal(mat_dict[("A", "B")], ref_hb_graph_chains['inter'], decimal=1)
    split_matrix = split_dict(mat_dict)
    # Ensure both matrices are of equal size and values
    assert_almost_equal(mat_dict["all"], sum(split_matrix), decimal=1)
    # Boundary test
    for i in range(len(split_matrix) - 1):
        common = np.logical_and(split_matrix[i] > 0, split_matrix[i+1] > 0)
        assert(common.sum() == 0)


