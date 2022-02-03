# Standard library
import os
import pkg_resources
# Third-party packages
import MDAnalysis as mda
import numpy as np
from numpy.testing import (
    assert_equal,
    assert_almost_equal)
import pytest
# libinteract
from libinteract import libinteract as li



########################## HELPER FUNCTIONS ###########################



def chaindict2list(dictionary):

    # Convert the dictionary to a list containing all tables/matrices
    # apart from those containing all contacts
    not_all_items = \
        [dictionary[k] for k in dictionary.keys() if k != "all"]

    # Get only the table/matrix containing all contacts
    all_item = dictionary["all"]

    # Return both
    return all_item, not_all_items


def dict_tables_test(groups2keys,
                     dict_tables,
                     ref_tables):

    # Check the equality of the contents of the tables
    # for the first 5 decimals
    for chain, key in groups2keys.items():
        for i, t in enumerate(dict_tables[chain]):
            assert(",".join(str(x) for x in t)[:5] == \
                   ref_tables[key][i].strip()[:5])

    # Get the table containing all the contacts and sort it
    all_contacts_table, not_all_contacts_tables = \
        chaindict2list(dict_tables)

    # Ensure that the sum of all the other tables is
    # almost equal to the values reported in the tables
    # containing all the contacts
    assert(np.array_equal(\
            np.sort(all_contacts_table, axis = 0), 
            np.sort(np.vstack(not_all_contacts_tables), axis = 0)) \
           == True)


def dict_matrices_test(groups2keys,
                       dict_matrices,
                       ref_matrices):
    
    # For each contacts' group/key corresponding to
    # the reference matrix
    for group, key in groups2keys.items():
        # Ensure that they are close enough
        assert_almost_equal(dict_matrices[group],
                            ref_matrices[key],
                            decimal = 1)
    
    # Get both the matrix containing all contacts and all the others
    all_contacts_matrix, not_all_contacts_matrices = \
        chaindict2list(dict_matrices)
    
    # Ensure that the sum of all the other matrices is
    # almost equal to the values reported in the matrix
    # containing all the contacts
    assert_almost_equal(all_contacts_matrix,
                        sum(not_all_contacts_matrices),
                        decimal = 1)
    
    # Boundary test
    for i in range(len(not_all_contacts_matrices) - 1):
        common = np.logical_and(not_all_contacts_matrices[i] > 0,
                                not_all_contacts_matrices[i+1] > 0)
        assert(common.sum() == 0)



############################## FIXTURES ###############################



#---------------------------- Directories ----------------------------#


@pytest.fixture
def ref_dirs(request):
    
    # Directories containing the references (expected results)
    return {"one_chain" : os.path.join(request.fspath.dirname, 
                                       "data/single_chain"),
            "two_chains" : os.path.join(request.fspath.dirname,
                                        "data/two_chains")}



#----------------------------- Input files ---------------------------#



@pytest.fixture
def systems_data(ref_dirs):
    
    # Directories containing expected results for the systems with
    # one chain or two chains
    one_chain = ref_dirs["one_chain"]
    two_chains = ref_dirs["two_chains"]

    # Files describing the systems (topology, trajectory, reference) 
    return {"one_chain" :
                {"gro" : os.path.join(one_chain, "sim.prot.gro"),
                 "xtc" : os.path.join(one_chain, "traj.xtc"),
                 "pdb" : os.path.join(one_chain, "sim.prot.A.pdb")},
            "two_chains" :
                {"gro" : os.path.join(two_chains, 
                                      "sim.prot.twochains.gro"),
                 "xtc" : os.path.join(two_chains, 
                                      "traj.twochains.xtc"),
                 "pdb" : os.path.join(two_chains, 
                                      "sim.prot.twochains.pdb")}}


@pytest.fixture
def systems(systems_data):

    # MDAnalysis Universe objects describing the systems
    return \
        {"one_chain" :
            {"uni" :
                mda.Universe(systems_data["one_chain"]["gro"],
                             systems_data["one_chain"]["xtc"]),
            "pdb" :
                mda.Universe(systems_data["one_chain"]["pdb"])},
         "two_chains" :
            {"uni" :
                mda.Universe(systems_data["two_chains"]["gro"],
                             systems_data["two_chains"]["xtc"]),
             "pdb" :     
                mda.Universe(systems_data["two_chains"]["pdb"])}}


@pytest.fixture
def acpsn_nf_file():

    # File containing the normalization factors to be
    # used for calculating the acPSN
    return pkg_resources.resource_filename("pyinteraph",
                                           "normalization_factors.ini")


@pytest.fixture
def sb_cg_file():
    
    # File containing the charged groups to be used fo
    # calculating electrostatic interactions
    return pkg_resources.resource_filename("pyinteraph",
                                           "charged_groups.ini")


@pytest.fixture
def hb_don_acc_file():
    
    # File containing the donor and acceptor atoms to be
    # used for calculating hydrogen bonds
    return pkg_resources.resource_filename("pyinteraph",
                                           "hydrogen_bonds.ini")


@pytest.fixture
def kbp_potential_file():
    
    # Binary file for the calculation of the statistical potential
    return pkg_resources.resource_filename("pyinteraph",
                                           "ff.S050.bin64")


@pytest.fixture
def kbp_atoms_file():
    
    # File containing the atoms (for each residue)
    # to be used to calculate the statistical potential
    return pkg_resources.resource_filename("pyinteraph",
                                           "kbp_atomlist")


@pytest.fixture
def masses_file():
    
    # File containing the parameters for atoms' masses (charmm27)
    return os.path.join(pkg_resources.resource_filename("pyinteraph",
                                                        "ff_masses"), 
                        "charmm27")


@pytest.fixture
def acpsn_normalization_factors(acpsn_nf_file):

    # Normalization factors to calculate the acPSN
    return li.parse_nf_file(acpsn_nf_file)


@pytest.fixture
def sb_charged_groups(sb_cg_file):
    
    # Charged groups to calculate the electrostatic interactions
    return li.parse_cgs_file(sb_cg_file)

@pytest.fixture
def hb_don_acc(hb_don_acc_file):
    
    # Donors and acceptors to calculate the hydrogen bonds
    return li.parse_hbs_file(hb_don_acc_file)


@pytest.fixture
def kbp_atomlist(kbp_atoms_file):
    
    # Atoms to calculate the statistical potential
    return li.parse_atomlist(kbp_atoms_file)



#----------------------------- References ----------------------------#
 


@pytest.fixture
def acpsn_references(ref_dirs):

    # Directories containing expected results for the system with
    # one chain or two chains
    oc = ref_dirs["one_chain"]
    tc = ref_dirs["two_chains"]

    # Names of the tables and matrices
    table = "acpsn"
    matrix = "acpsn-graph"

    # Suffixes pertaining to the files of the two-chains system
    # (all contacts, intrachain and intrachain contacts)
    tc_all = "twochains_all"
    tc_a = "twochains_intra_A"
    tc_b = "twochains_intra_B"
    tc_ab = "twochains_inter_A-B"

    # Expected results for the acPSN
    return \
        {"tables" :
          {"one_chain" : open(f"{oc}/{table}.csv").readlines(),
         "two_chains" :
           {"all" : open(f"{tc}/{table}_{tc_all}.csv").readlines(),
            "intra_a" : open(f"{tc}/{table}_{tc_a}.csv").readlines(),
            "intra_b" : open(f"{tc}/{table}_{tc_b}.csv").readlines(),
            "inter_ab" : open(f"{tc}/{table}_{tc_ab}.csv").readlines()}},
         "matrices" :
            {"one_chain" : np.loadtxt(f"{oc}/{matrix}.dat"),
             "two_chains" :
               {"all" : np.loadtxt(f"{tc}/{matrix}_{tc_all}.dat"),
                "intra_a" : np.loadtxt(f"{tc}/{matrix}_{tc_a}.dat"),
                "intra_b" : np.loadtxt(f"{tc}/{matrix}_{tc_b}.dat"),
                "inter_ab" : np.loadtxt(f"{tc}/{matrix}_{tc_ab}.dat")}}}


@pytest.fixture
def hc_references(ref_dirs):

    # Directories containing expected results for the system with
    # one chain or two chains
    oc = ref_dirs["one_chain"]
    tc = ref_dirs["two_chains"]

    # Suffix for files pertaining to the rg-corrected hydrophobic
    # contacts
    rg = "rg_corrected_all"

    # Names of the tables and matrices
    table = "hydrophobic-clusters"
    matrix = "hc-graph"

    # Suffixes pertaining to the files of the two-chains system
    # (all contacts, intrachain and intrachain contacts)
    tc_all = "twochains_all"
    tc_a = "twochains_intra_A"
    tc_ab = "twochains_inter_A-B"

    # Expected results for the hydrophobic contacts
    return \
        {"tables" :
          {"one_chain" :
            {"null" : open(f"{oc}/{table}.csv").readlines(),
             "rg" : open(f"{oc}/{table}_{rg}.csv").readlines()},
         "two_chains" :
           {"null" :
             {"all" : open(f"{tc}/{table}_{tc_all}.csv").readlines(),
              "intra_a" : open(f"{tc}/{table}_{tc_a}.csv").readlines(),
              "inter_ab" : open(f"{tc}/{table}_{tc_ab}.csv").readlines()}}},
         "matrices" :
            {"one_chain" :
                {"null" : np.loadtxt(f"{oc}/{matrix}.dat"),
                 "rg" : np.loadtxt(f"{oc}/{matrix}_{rg}.dat")},
             "two_chains" :
               {"null" :
                 {"all" : np.loadtxt(f"{tc}/{matrix}_{tc_all}.dat"),
                  "intra_a" : np.loadtxt(f"{tc}/{matrix}_{tc_a}.dat"),
                  "inter_ab" : np.loadtxt(f"{tc}/{matrix}_{tc_ab}.dat")}}}}


@pytest.fixture
def sb_references(ref_dirs):

    # Directories containing expected results for the system with
    # one chain or two chains
    oc = ref_dirs["one_chain"]
    tc = ref_dirs["two_chains"]

    # Names of the tables and matrices
    table = "salt-bridges"
    matrix = "sb-graph"

    # Suffixes pertaining to the files of the two-chains system
    # (all contacts, intrachain and intrachain contacts)
    tc_all = "twochains_all"
    tc_a = "twochains_intra_A"
    tc_ab = "twochains_inter_A-B"

    # Expected results for the salt bridges
    return \
        {"tables" :
          {"one_chain" : open(f"{oc}/{table}.csv").readlines(),
           "two_chains" :
            {"all" : open(f"{tc}/{table}_{tc_all}.csv").readlines(),
             "intra_a" : open(f"{tc}/{table}_{tc_a}.csv").readlines(),
             "inter_ab" : open(f"{tc}/{table}_{tc_ab}.csv").readlines()}},
         "matrices" :
            {"one_chain" : np.loadtxt(f"{oc}/{matrix}.dat"),
             "two_chains" :
               {"all" : np.loadtxt(f"{tc}/{matrix}_{tc_all}.dat"),
                "intra_a" : np.loadtxt(f"{tc}/{matrix}_{tc_a}.dat"),
                "inter_ab" : np.loadtxt(f"{tc}/{matrix}_{tc_ab}.dat")}}}


@pytest.fixture
def hb_references(ref_dirs):

    # Directories containing expected results for the system with
    # one chain or two chains
    oc = ref_dirs["one_chain"]
    tc = ref_dirs["two_chains"]

    # Names of the tables and matrices
    table = "hydrogen-bonds"
    matrix = "hb-graph"

    # Suffixes pertaining to the files of the two-chains system
    # (all contacts, intrachain and intrachain contacts)
    tc_all = "twochains_all"
    tc_a = "twochains_intra_A"
    tc_b = "twochains_intra_B"
    tc_ab = "twochains_inter_A-B"

    # Expected results for the hydrogen bonds
    return \
        {"tables" :
          {"one_chain" : open(f"{oc}/{table}.csv").readlines(),
         "two_chains" :
           {"all" : open(f"{tc}/{table}_{tc_all}.csv").readlines(),
            "intra_a" : open(f"{tc}/{table}_{tc_a}.csv").readlines(),
            "intra_b" : open(f"{tc}/{table}_{tc_b}.csv").readlines(),
            "inter_ab" : open(f"{tc}/{table}_{tc_ab}.csv").readlines()}},
         "matrices" :
            {"one_chain" : np.loadtxt(f"{oc}/{matrix}.dat"),
             "two_chains" :
               {"all" : np.loadtxt(f"{tc}/{matrix}_{tc_all}.dat"),
                "intra_a" : np.loadtxt(f"{tc}/{matrix}_{tc_a}.dat"),
                "intra_b" : np.loadtxt(f"{tc}/{matrix}_{tc_b}.dat"),
                "inter_ab" : np.loadtxt(f"{tc}/{matrix}_{tc_ab}.dat")}}}


@pytest.fixture
def kbp_references(ref_dirs):

    # Directories containing expected results for the system with
    # one chain
    oc = ref_dirs["one_chain"]

    # Names of the string and matrix
    table = "kbp-table"
    matrix = "kbp-graph"

    # Expected results for the statistical potential
    return \
        {"tables" :
            {"one_chain" : open(f"{oc}/{table}.csv").readlines()},
         "matrices" :
            {"one_chain" : np.loadtxt(f"{oc}/{matrix}.dat")}}



#-------------------------- Residues' lists --------------------------#



@pytest.fixture
def hc_residues_list():
    
    # Residues to be used to calculate hydrophobic contacts
    return ["ALA", "VAL", "LEU", "ILE",
            "PHE", "PRO", "TRP", "MET"]


@pytest.fixture
def sc_residues_list():
    
    # Residues to be used to calculate side-chain contacts (used for
    # both the acPSN and the statistical potential)
    return ["ALA", "ARG", "ASN", "ASP",
            "CYS", "GLN", "GLU", "HIS",
            "ILE", "LEU", "LYS", "MET",
            "PHE", "PRO", "SER", "THR",
            "TRP", "TYR", "VAL"]


@pytest.fixture
def hb_selections():
    
    # Selections to be used to calculate hydrogen bonds
    return {"protein_protein" :
                {"sel1" : "protein", 
                 "sel2" : "protein"}}



#------------------------------- Sparse ------------------------------#



@pytest.fixture
def sparse_list():
    return np.arange(0, 10)


@pytest.fixture
def sparse_obj(sparse_list):
    return li.Sparse(sparse_list)


@pytest.fixture
def sparse_bin():
    return ["a", "b", "c", "d", 1]



#----------------------------- Parameters ----------------------------#



@pytest.fixture
def acpsn_parameters(acpsn_normalization_factors):
    
    # General parameters for the calculation of the acPSN
    return {"co" : 4.5,
            "perco" : 0.0,
            "proxco" : 1,
            "imin" : 3.0,
            "edge_weight" : "strength",
            "norm_facts" : acpsn_normalization_factors,
            "nf_permissive" : False,
            "nf_default" : 999.9}


@pytest.fixture
def hc_parameters(hc_residues_list, sc_residues_list):

    # General parameters for the calculation of hydrophobic contacts
    parameters = {"identfunc" : li.generate_sc_identifiers,
                  "co" : 5.0,
                  "perco" : 0.0,
                  "ffmasses" : "charmm27",
                  "fullmatrixfunc" : li.calc_sc_fullmatrix,
                  "mindist" : False}

    # Specific parameters for the correction to be applied
    return {"null" :
                {"correction_func" : li.null_correction,
                 "reslist" : hc_residues_list,
                 **parameters},
            "rg" :
                {"correction_func" : li.rg_correction,
                 "reslist" : sc_residues_list,
                 **parameters}}


@pytest.fixture
def sb_parameters(sb_charged_groups):
    
    # General parameters for the calculation of salt bridges
    return {"identfunc" : li.generate_cg_identifiers,
            "co" : 4.5,
            "perco" : 0,
            "ffmasses" : "charmm27",
            "fullmatrixfunc" : li.calc_cg_fullmatrix,
            "mindist" : True,
            "mindist_mode" : "diff",
            "cgs" : sb_charged_groups}


@pytest.fixture
def hb_parameters(hb_don_acc):

    # General parameters for the calculation of hydrogen bonds
    parameters = {"distance" : 3.5,
                  "angle" : 120.0,
                  "perco" : 0.0,
                  "perresidue" : False,
                  "do_fullmatrix" : True,
                  "other_hbs" : hb_don_acc}

    # Specific parameters for hydrogen bonds calculated between
    # all protein residues
    return {"protein_protein" :
                {"sel1" : "protein",
                 "sel2" : "protein",
                 **parameters}}


@pytest.fixture
def kbp_parameters(kbp_atomlist,
                   sc_residues_list,
                   kbp_potential_file):
    
    # Parameters for the calculation of the statistical potential
    return {"kbp_atomlist" : kbp_atomlist,
            "residues_list" : sc_residues_list,
            "potential_file" : kbp_potential_file,
            "do_fullmatrix" : True}



#----------------------------- Contacts ------------------------------# 



@pytest.fixture
def acpsn_twochains(systems,
                    acpsn_parameters):
    
    # Calculate the acPSN
    table, matrix = li.do_acpsn(pdb = systems["two_chains"]["pdb"],
                                uni = systems["two_chains"]["uni"],
                                **acpsn_parameters)

    # Return the table of contacts and the matrix
    return {"table" : table, "matrix" : matrix}


@pytest.fixture
def hc_twochains(systems,
                 hc_parameters):

    # Calculate the hydrophobic contacts   
    table, matrix = li.do_interact(pdb = systems["two_chains"]["pdb"],
                                   uni = systems["two_chains"]["uni"],
                                   **hc_parameters["null"])

    # Return the table of contacts and the matrix
    return {"table" : table, "matrix" : matrix}


@pytest.fixture
def sb_twochains(systems,
                 sb_parameters):
    
    # Calculate the salt bridges
    table, matrix = li.do_interact(pdb = systems["two_chains"]["pdb"],
                                   uni = systems["two_chains"]["uni"],
                                   **sb_parameters)

    # Return the table of contacts and the matrix
    return {"table" : table, "matrix" : matrix}


@pytest.fixture
def hb_twochains(systems,
                 hb_parameters):
    
    # Calculate the hydrogen bonds
    table, matrix = li.do_hbonds(pdb = systems["two_chains"]["pdb"],
                                 uni = systems["two_chains"]["uni"],
                                 **hb_parameters["protein_protein"])

    # Return the table of contacts and the matrix
    return {"table" : table, "matrix" : matrix}



############################### TESTS #################################



#--------------------------- File parsing ----------------------------#



def test_parse_nf_file(acpsn_nf_file):

    # Test the parsing of the normalization factors file
    li.parse_nf_file(acpsn_nf_file)


def test_parse_cg_file(sb_cg_file):

    # Test the parsing of the charged groups file
    li.parse_cgs_file(sb_cg_file)


def test_parse_hb_file(hb_don_acc_file):

    # Test the parsing of the donors/acceptors file 
    li.parse_hbs_file(hb_don_acc_file)


def test_parse_sparse(kbp_potential_file):

    # Test the parsing of the statistical potential file
    li.parse_sparse(kbp_potential_file)


def test_parse_atomlist(kbp_atoms_file):

    # Test the parsing of the atom list
    data = li.parse_atomlist(kbp_atoms_file)

    # Check that the data parsed are correct
    assert(len(data) == 20)
    assert(data["GLN"] == ["N", "CA", "C", "O", "CB", 
                           "CG", "CD", "OE1", "NE2"])



#----------------------- Identifiers generation ----------------------#



def test_generate_cg_identifiers(systems,
                                 sb_charged_groups):
    
    # Test the generation of charged groups' identifiers
    li.generate_cg_identifiers(pdb = systems["one_chain"]["pdb"],
                               uni = systems["one_chain"]["uni"],
                               cgs = sb_charged_groups)


def test_generate_sc_identifiers(systems,
                                 hc_residues_list):
    
    # Test the generation of side chains' identifiers    
    li.generate_sc_identifiers(pdb = systems["one_chain"]["pdb"],
                               uni = systems["one_chain"]["uni"],
                               reslist = hc_residues_list)



#------------------------------- Sparse ------------------------------#



def test_sparse_constructor(sparse_list,
                            sparse_obj):
    
    data = np.array([sparse_obj.r1, sparse_obj.r2,
                     sparse_obj.p1_1, sparse_obj.p1_2,
                     sparse_obj.p2_1, sparse_obj.p2_2,
                     sparse_obj.cutoff**2,
                     1.0 / sparse_obj.step,
                     sparse_obj.total,
                     sparse_obj.num])
        
    assert_almost_equal(data, sparse_list, decimal = 10)


def test_add_bin(sparse_obj,
                 sparse_bin):
    
    sparse_obj.add_bin(sparse_bin)

    key = "".join(sparse_bin[0:4])
    val = sparse_bin[4]

    assert_equal(sparse_obj.bins[key], val)


def test_num_bins(sparse_obj,
                  sparse_bin):
    
    sparse_obj.add_bin(sparse_bin)

    assert_equal(len(sparse_obj.bins), 1)


def test_ff_masses(systems,
                   masses_file):
    
    selstr = "resid 10 and not backbone"
    sel = [systems["one_chain"]["uni"].select_atoms(selstr)]
    li.assign_ff_masses(masses_file, sel)



#------------------------------- acPSN -------------------------------#



def test_do_acpsn(systems,
                  acpsn_parameters,
                  acpsn_references):
    
    # Get the reference table and the reference matrix
    ref_table = acpsn_references["tables"]["one_chain"]
    ref_matrix = acpsn_references["matrices"]["one_chain"]
    
    # Compute the table and the matrix
    table, matrix = \
        li.do_acpsn(pdb = systems["one_chain"]["pdb"],
                    uni = systems["one_chain"]["uni"],
                    **acpsn_parameters)
    
    # Test the equality between the matrix computed and the reference
    assert_almost_equal(matrix, ref_matrix, decimal = 1)

    # Test the equality between the table computed and the reference
    for i, t in enumerate(table):
        assert(",".join(str(x) for x in t) == ref_table[i].strip())


def test_create_dict_tables_acpsn(acpsn_twochains,
                                  acpsn_references):
    
    # Get the dictionary of reference tables
    ref_tables = acpsn_references["tables"]["two_chains"]

    # Get the dictionary containing the computed tables
    dict_tables = li.create_dict_tables(acpsn_twochains["table"])

    # Dictionary mapping the groups of contacts (all, intrachain,
    # interchain) to the corresponding keys in the dictionary of
    # reference tables
    groups2keys = {"all" : "all", 
                   "A" : "intra_a",
                   "B" : "intra_b",
                   ("A", "B") : "inter_ab"}
    
    # Check the computed tables against the references
    dict_tables_test(groups2keys, dict_tables, ref_tables)


def test_create_dict_matrices_acpsn(systems,
                                    acpsn_twochains,
                                    acpsn_references):
    
    # Get the reference matrices
    ref_matrices = acpsn_references["matrices"]["two_chains"]

    # Get the dictionary containing the computed tables
    dict_tables = \
        li.create_dict_tables(acpsn_twochains["table"])
    
    # Get the dictionary containing the computed matrices    
    dict_matrices = \
        li.create_dict_matrices(acpsn_twochains["matrix"],
                                dict_tables,
                                systems["two_chains"]["pdb"])

    # Dictionary mapping the groups of contacts (all, intrachain,
    # interchain) to the corresponding keys in the dictionary of
    # reference matrices
    groups2keys = {"all" : "all", 
                   "A" : "intra_a",
                   "B" : "intra_b",
                   ("A", "B") : "inter_ab"}

    # Check the computed matrices against the references
    dict_matrices_test(groups2keys, dict_matrices, ref_matrices)



#------------------------ Hydrophobic contacts -----------------------#



def test_do_hc(systems,
               hc_parameters,
               hc_references):
    
    # Get the reference table and the reference matrix
    ref_table = hc_references["tables"]["one_chain"]["null"]
    ref_matrix = hc_references["matrices"]["one_chain"]["null"]
    
    # Compute the table and the matrix
    table, matrix = \
        li.do_interact(pdb = systems["one_chain"]["pdb"],
                       uni = systems["one_chain"]["uni"],
                       **hc_parameters["null"])
    
    # Test the equality between the matrix computed and the reference
    assert_almost_equal(matrix, ref_matrix, decimal = 1)

    # Test the equality between the table computed and the reference
    for i, t in enumerate(table):
        assert(",".join(str(x) for x in t) == ref_table[i].strip())


def test_do_hc_rg(systems,
                  hc_parameters,
                  hc_references):

    # Get the reference table and the reference matrix
    ref_table = hc_references["tables"]["one_chain"]["rg"]
    ref_matrix = hc_references["matrices"]["one_chain"]["rg"]

    # Compute the table and the matrix
    table, matrix = \
        li.do_interact(pdb = systems["one_chain"]["pdb"],
                       uni = systems["one_chain"]["uni"],
                       **hc_parameters["rg"])

    # Test the equality between the matrix computed and the reference
    assert_almost_equal(matrix, ref_matrix, decimal = 1)

    # Test the equality between the table computed and the reference
    for i, t in enumerate(table):
        assert(",".join(str(x) for x in t) == ref_table[i].strip())


def test_create_dict_tables_hc(hc_twochains,
                               hc_references):
    
    # Get the dictionary of reference tables
    ref_tables = hc_references["tables"]["two_chains"]["null"]

    # Get the dictionary containing the computed tables
    dict_tables = li.create_dict_tables(hc_twochains["table"])

    # Dictionary mapping the groups of contacts (all, intrachain,
    # interchain) to the corresponding keys in the dictionary of
    # reference tables
    groups2keys = {"all" : "all", 
                   "A" : "intra_a",
                   ("A", "B") : "inter_ab"}

    # Check the computed tables against the references
    dict_tables_test(groups2keys, dict_tables, ref_tables)


def test_create_dict_matrices_hc(systems,
                                 hc_twochains,
                                 hc_references):
    
    # Get the reference matrices
    ref_matrices = hc_references["matrices"]["two_chains"]["null"]

    # Get the dictionary containing the computed tables
    dict_tables = \
        li.create_dict_tables(hc_twochains["table"])
    
    # Get the dictionary containing the computed matrices    
    dict_matrices = \
        li.create_dict_matrices(hc_twochains["matrix"],
                                dict_tables,
                                systems["two_chains"]["pdb"])

    # Dictionary mapping the groups of contacts (all, intrachain,
    # interchain) to the corresponding keys in the dictionary of
    # reference matrices
    groups2keys = {"all" : "all", 
                   "A" : "intra_a",
                   ("A", "B") : "inter_ab"}

    # Check the computed matrices against the references
    dict_matrices_test(groups2keys, dict_matrices, ref_matrices)



#--------------------------- Salt bridges ----------------------------#



def test_do_sb(systems,
               sb_parameters,
               sb_references):

    # Get the reference table and the reference matrix
    ref_table = sb_references["tables"]["one_chain"]
    ref_matrix = sb_references["matrices"]["one_chain"]

    # Compute the table and the matrix
    table, matrix = \
        li.do_interact(pdb = systems["one_chain"]["pdb"],
                       uni = systems["one_chain"]["uni"],
                       **sb_parameters)

    # Test the equality between the matrix computed and the reference
    assert_almost_equal(matrix, ref_matrix, decimal = 1)

    # Test the equality between the table computed and the reference
    for i, t in enumerate(table):
        assert(",".join(str(x) for x in t) == ref_table[i].strip())


def test_create_dict_tables_sb(sb_twochains,
                               sb_references):

    # Get the reference tables
    ref_tables = sb_references["tables"]["two_chains"]

    # Get the dictionary containing the computed tables
    dict_tables = li.create_dict_tables(sb_twochains["table"])

    # Dictionary mapping the groups of contacts (all, intrachain,
    # interchain) to the corresponding keys in the dictionary of
    # reference tables
    groups2keys = {"all" : "all", 
                   "A" : "intra_a",
                   ("A", "B") : "inter_ab"}

    # Check the computed tables against the references
    dict_tables_test(groups2keys, dict_tables, ref_tables)


def test_create_dict_matrices_sb(systems,
                                 sb_twochains,
                                 sb_references):
    
    # Get the reference matrices 
    ref_matrices = sb_references["matrices"]["two_chains"]

    # Get the dictionary containing the computed tables    
    dict_tables = \
        li.create_dict_tables(sb_twochains["table"])

    # Get the dictionary containing the computed matrices  
    dict_matrices = \
        li.create_dict_matrices(sb_twochains["matrix"],
                                dict_tables,
                                systems["two_chains"]["pdb"])

    # Dictionary mapping the groups of contacts (all, intrachain,
    # interchain) to the corresponding keys in the dictionary of
    # reference matrices
    groups2keys = {"all" : "all", 
                   "A" : "intra_a",
                   ("A", "B") : "inter_ab"}

    # Check the computed matrices against the references
    dict_matrices_test(groups2keys, dict_matrices, ref_matrices)



#-------------------------- Hydrogen bonds ---------------------------#



def test_do_hb(systems,
               hb_parameters,
               hb_references):

    # Get the reference table and the reference matrix
    ref_table = hb_references["tables"]["one_chain"]
    ref_matrix = hb_references["matrices"]["one_chain"]

    # Compute the table and the matrix
    table, matrix = \
        li.do_hbonds(pdb = systems["one_chain"]["pdb"],
                     uni = systems["one_chain"]["uni"],
                     **hb_parameters["protein_protein"])

    # Test the equality between the matrix computed and the reference
    assert_almost_equal(matrix, ref_matrix, decimal = 1)

    # Test the equality between the table computed and the reference
    for i, t in enumerate(table):
        assert(",".join(str(x) for x in t) == ref_table[i].strip())


def test_create_dict_tables_hb(hb_twochains,
                              hb_references):

    # Get the reference tables
    ref_tables = hb_references["tables"]["two_chains"]

    # Get the dictionary containing the computed tables
    dict_tables = li.create_dict_tables(hb_twochains["table"])

    # Dictionary mapping the groups of contacts (all, intrachain,
    # interchain) to the corresponding keys in the dictionary of
    # reference tables
    groups2keys = {"all" : "all", 
                   "A" : "intra_a",
                   "B" : "intra_b",
                   ("A", "B") : "inter_ab"}

    # Check the equality of the contents of the tables
    dict_tables_test(groups2keys, dict_tables, ref_tables)


def test_create_dict_matrices_hb(systems,
                                 hb_twochains,
                                 hb_references):

    # Get the reference matrices
    ref_matrices = hb_references["matrices"]["two_chains"]

    # Get the dictionary containing the computed tables    
    dict_tables = \
        li.create_dict_tables(hb_twochains["table"])

    # Get the dictionary containing the computed matrices 
    dict_matrices = \
        li.create_dict_matrices(hb_twochains["matrix"],
                                dict_tables,
                                systems["two_chains"]["pdb"])
    
    # Dictionary mapping the groups of contacts (all, intrachain,
    # interchain) to the corresponding keys in the dictionary of
    # reference matrices
    groups2keys = {"all" : "all", 
                   "A" : "intra_a",
                   "B" : "intra_b",
                   ("A", "B") : "inter_ab"}

    # Check the computed matrices against the references
    dict_matrices_test(groups2keys, dict_matrices, ref_matrices)



#----------------------- Statistical potential -----------------------#



def test_do_potential(systems,
                      kbp_parameters,
                      kbp_references):
    
    # Get the reference table and the reference matrix
    ref_table = kbp_references["tables"]["one_chain"]
    ref_matrix = kbp_references["matrices"]["one_chain"]

    # Get the computed table and matrix
    table, matrix = \
        li.do_potential(uni = systems["one_chain"]["uni"],
                        pdb = systems["one_chain"]["pdb"],
                        **kbp_parameters)

    # Check the equality between the computed matrix and the
    # reference matrix    
    assert_almost_equal(matrix, ref_matrix, decimal = 3)

    # Test the equality between the table computed and the reference
    for i, t in enumerate(table):
        assert(",".join(str(x) for x in t) == ref_table[i].strip())
