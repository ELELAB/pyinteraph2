# Standard library
import os
# Third-party packages
import MDAnalysis as mda
import networkx as nx
import numpy as np
from numpy.testing import (
    assert_equal,
    assert_almost_equal)
import pandas as pd
import pytest
# libinteract
from pyinteraph import centrality_analysis as ca
from pyinteraph import path_analysis as pa




########################## HELPER FUNCTIONS ###########################



def centrality_test(measure, centrality_values, ref_values):

    # Test the almost-equality of a dictionary to another
    for item, value in centrality_values.items():
        assert_almost_equal(value, ref_values[measure][item])



############################## FIXTURES ###############################



#---------------------------- Directories ----------------------------#



@pytest.fixture
def ref_dirs(request):
    
    # Directories containing the references (expected results)
    return {"two_chains" : os.path.join(request.fspath.dirname, 
                                        "data/two_chains")}



#-------------------------------- Inputs -----------------------------#



@pytest.fixture
def data_files(ref_dirs):

    # Directory containing expected results for the system with
    # two chains
    one_chain = ref_dirs["two_chains"]

    # Data files
    return {"two_chains" : 
                {"ref" : os.path.join(one_chain, 
                                      "sim.prot.twochains.pdb"),
                 "matrix" : os.path.join(one_chain, 
                                        "sc-graph_twochains_all.dat")}}


@pytest.fixture
def graph(data_files):

    # Graph built from the input data files
    return pa.build_graph(data_files["two_chains"]["matrix"],
                          data_files["two_chains"]["ref"])


@pytest.fixture
def key_args(graph):

    # Keyword arguments to be used when performing the
    # centrality analyses
    return {"weight" : None,
            "normalized" : True,
            "endpoints" : False,
            "max_iter" : 100,
            "tol" : 1e-06,
            "hub" : 3,
            "identifiers" : graph[0],
            "residue_names" : graph[1]}


@pytest.fixture
def centrality_list():

    # List of centrality measures to be calculated
    return ["hubs", "degree", "betweenness", "closeness",
            "eigenvector", "current_flow_betweenness",
            "current_flow_closeness", "edge_betweenness",
            "edge_current_flow_betweenness"]


@pytest.fixture
def G():

    # Edges of the graph
    edges = \
        {("1", "2"), ("2", "4"), ("1", "3"), ("3", "4"), ("4", "5"),
         ("6", "7"), ("7", "8"), ("6", "9"), ("9", "8"), ("5", "6")}
    
    # Create an empty graph
    G = nx.Graph()

    # Add the edges
    G.add_edges_from(edges)

    # Return the graph
    return G


#----------------------------- References ----------------------------#



@pytest.fixture
def ref_files(ref_dirs):

    # Directory containing the reference failes for the two-chains
    # system
    two_chains = ref_dirs["two_chains"]
    
    # Reference files containing the centrality results
    return {"two_chains" : \
                {"node" : os.path.join(two_chains, 
                                       "centrality_node.csv"),
                 "edge" : os.path.join(two_chains,
                                       "centrality_edge.csv")}}


@pytest.fixture
def ref_values():

    # Reference values for the centrality measures
    return \
        {"hubs" : \
            {"1" : 0, "2" : 0, "3" : 0, "4" : 3, "5" : 0,
             "6" : 3, "7" : 0, "8" : 0, "9" : 0},
         "degree" : \
            {"1" : 0.25, "2" : 0.25, "3" : 0.25, "4" : 0.375,
             "5" : 0.25, "6" : 0.375, "7" : 0.25, "8" : 0.25, 
             "9" : 0.25},
         "betweenness" : \
            {"1" : 0.017857142857142856, "2" : 0.10714285714285714,
             "3" : 0.10714285714285714, "4" : 0.5535714285714285, 
             "5" : 0.5714285714285714, "6" : 0.5535714285714285, 
             "7" : 0.10714285714285714, "8" : 0.017857142857142856,
             "9" : 0.10714285714285714},
         "closeness" : \
            {"1" : 0.2962962962962963, "2" : 0.36363636363636365, 
             "3" : 0.36363636363636365,"4" : 0.47058823529411764, 
             "5" : 0.5, "6" : 0.47058823529411764, 
             "7" : 0.36363636363636365, "8" : 0.2962962962962963, 
             "9" : 0.36363636363636365},
         "eigenvector" : \
             {"1" : 0.2628655560595658, "2" : 0.300750477503772, 
              "3" : 0.30075047750377204, "4" : 0.4253254041760193, 
              "5" : 0.3717480344601847, "6" : 0.4253254041760208, 
              "7" : 0.30075047750377376, "8" : 0.2628655560595674, 
              "9" : 0.30075047750377376}, \
         "current_flow_betweenness" : \
             {"1" : 0.125, "2" : 0.16964285714285715, 
              "3" : 0.16964285714285715, "4":  0.5714285714285714,  
              "5" : 0.5714285714285714, "6" : 0.5714285714285715, 
              "7" : 0.16964285714285715, "8" : 0.12500000000000025,
              "9" : 0.169642857142857},
         "current_flow_closeness" : \
             {"1" : 0.05263157894736844, "2" : 0.05633802816901411, 
              "3" : 0.05633802816901411, "4" : 0.07142857142857145, 
              "5" : 0.07692307692307693, "6" : 0.07142857142857148, 
              "7" : 0.05633802816901411, "8" : 0.05263157894736844, 
              "9" : 0.05633802816901407}, \
         "edge_betweenness" : \
             {"1,2" : 0.125, "1,3" : 0.125, "2,4" : 0.2638888888888889,
              "3,4" : 0.2638888888888889,"4,5" : 0.5555555555555556,
              "5,6" : 0.5555555555555556, "6,7" : 0.2638888888888889, 
              "6,9": 0.2638888888888889, "7,8" : 0.125, "8,9" : 0.125},
         "edge_current_flow_betweenness" : \
             {"1,2" : 0.13392857142857142, "1,3" : 0.13392857142857142,
              "2,4" : 0.17857142857142858, "3,4" : 0.17857142857142858,
              "4,5" : 0.35714285714285715, "5,6" : 0.35714285714285715,
              "6,7" : 0.17857142857142863, "6,9" : 0.17857142857142858,
              "7,8" : 0.13392857142857145, "8,9" : 0.13392857142857145}}



############################### TESTS #################################



def test_get_hubs(G, ref_values, key_args):
    
    # Get the hubs
    values = ca.get_hubs(G, **key_args)
    # Check the values against the references
    for node, value in values.items():
        assert value == ref_values["hubs"][node]


def test_get_degree_centrality(G, ref_values, key_args):

    # Get the degree centrality values
    values = ca.get_degree_centrality(G, **key_args)
    # Check the values against the references
    centrality_test("degree", values, ref_values)


def test_get_betweenness_centrality(G, ref_values, key_args):

    # Get the betweenness centrality values
    values = ca.get_betweenness_centrality(G, **key_args)
    # Check the values against the references
    centrality_test("betweenness", values, ref_values)


def test_get_closeness_centrality(G, ref_values, key_args):

    # Get the closeness centrality values
    values = ca.get_closeness_centrality(G, **key_args)
    # Check the values against the references
    centrality_test("closeness", values, ref_values)


def test_get_eigenvector_centrality(G, ref_values, key_args):

    # Get the eigenvector centrality values
    values = ca.get_eigenvector_centrality(G, **key_args)
    # Check the values against the references
    centrality_test("eigenvector", values, ref_values)


def test_get_current_flow_betweenness_centrality(G,
                                                 ref_values,
                                                 key_args):

    # Get the current flow betweenness centrality values
    values = ca.get_current_flow_betweenness_centrality(G, **key_args)
    # Check the values against the references
    centrality_test("current_flow_betweenness", values, ref_values)


def test_get_current_flow_closeness_centrality(G,
                                               ref_values,
                                               key_args):

    # Get the current flow closeness centrality values
    values = ca.get_current_flow_closeness_centrality(G, **key_args)
    # Check the values against the references
    centrality_test("current_flow_closeness", values, ref_values)


def test_get_edge_betweenness_centrality(G, ref_values, key_args):

    # Get the edge betweenness centrality values
    values = ca.get_edge_betweenness_centrality(G, **key_args)
    # Check the values against the references
    centrality_test("edge_betweenness", values, ref_values)


def test_get_edge_current_flow_betweenness_centrality(G,
                                                      ref_values,
                                                      key_args):

    # Get the edge current flow betweenness centrality values
    values = \
        ca.get_edge_current_flow_betweenness_centrality(G, **key_args)
    # Check the values against the references
    centrality_test("edge_current_flow_betweenness", values, ref_values)


def test_get_centrality_dict(centrality_list, ref_files, graph, key_args):

    # Get the dictionary of centrality values for both node and
    # edge centralities
    node_dict, edge_dict = ca.get_centrality_dict(centrality_list,
                                                  ca.name2function,
                                                  graph[2],
                                                  **key_args)
    
    # Load the reference file for node centralities
    node_df = pd.read_csv(ref_files["two_chains"]["node"], sep = ",")
    
    # Check the values computed against the references
    for measure, val_dict in node_dict.items():
        values = [d for n, d in val_dict.items()]
        ref_values = list(node_df[measure])
        if measure == "node" or measure == "name":
            assert values == ref_values
        else:
            assert_almost_equal(values, ref_values)
    
    # Load the reference file for edge centralities
    edge_df = pd.read_csv(ref_files["two_chains"]["edge"], sep = ",")

    # Check the values computed against the references
    for measure, val_dict in edge_dict.items():
        values = [d for n, d in val_dict.items()]
        ref_values = list(edge_df[measure])
        if measure == "edge":
            assert values == ref_values
        else:
            assert_almost_equal(values, ref_values)
