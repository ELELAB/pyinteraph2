import os
import pytest
import networkx as nx
import pandas as pd
from numpy.testing import assert_almost_equal, assert_equal
from pyinteraph import centrality_analysis as ca
from pyinteraph import path_analysis as pa

# load files
@pytest.fixture
def ref_dir(request):
    return os.path.join(request.fspath.dirname, 'data/two_chains')

@pytest.fixture
def data_files(ref_dir):
    return { 
             'pdb' : os.path.join(ref_dir, 'sim.prot.twochains.pdb'),
             'psn' : os.path.join(ref_dir, 'sc-graph_twochains_all.dat')
           }

@pytest.fixture
def build_graph(data_files):
    return pa.build_graph(data_files['psn'], data_files['pdb'])

@pytest.fixture
def ref_name(ref_dir):
    return {
             'node' : os.path.join(ref_dir, 'centrality.txt'),
             'edge' : os.path.join(ref_dir, 'centrality_edge.txt')
           }

@pytest.fixture
def centrality_list():
    return ["hubs", "degree", "betweenness", "closeness", "eigenvector", 
            "current_flow_betweenness", "current_flow_closeness", 
            "edge_betweenness", "edge_current_flow_betweenness"]

@pytest.fixture
def kwargs(build_graph):
    return {"weight" : None,
              "normalized" : True,
              "endpoints" : False,
              "max_iter" : 100,
              "tol" : 1e-06,
              "hub" : 3,
              "identifiers" : build_graph[0],
              "residue_names" : build_graph[1]}

@pytest.fixture
def centrality_dict(centrality_list, build_graph, kwargs):
    G = build_graph[2]
    return ca.get_centrality_dict(centrality_list, ca.function_map, G, **kwargs)

@pytest.fixture
def G():
    edges = [('1', '2'), ('2', '4'), ('1', '3'), ('3', '4'), ('4', '5'),
             ('6', '7'), ('7', '8'), ('6', '9'), ('9', '8'), ('5', '6')]
    G = nx.Graph()
    G.add_edges_from(edges)
    return G

# tests
def test_get_hubs(G, kwargs):
    ref_values = {'1': 0, '2': 0, '3': 0, '4': 3, '5': 0,
                  '6': 3, '7': 0, '8': 0, '9': 0}
    val_dict = ca.get_hubs(G, **kwargs)
    for node, value in val_dict.items():
        assert value == ref_values[node]

def test_get_degree_cent(G, kwargs):
    ref_values = {'1': 0.25, '2': 0.25, '4': 0.375, '3': 0.25, '5': 0.25, 
                  '6': 0.375, '7': 0.25, '8': 0.25, '9': 0.25}
    val_dict = ca.get_degree_cent(G, **kwargs)
    for node, value in val_dict.items():
        assert_almost_equal(value, ref_values[node])

def test_get_betweeness_cent(G, kwargs):
    ref_values = {'1': 0.017857142857142856, '2': 0.10714285714285714, 
                  '4': 0.5535714285714285, '3': 0.10714285714285714, 
                  '5': 0.5714285714285714, '6': 0.5535714285714285, 
                  '7': 0.10714285714285714, '8': 0.017857142857142856, 
                  '9': 0.10714285714285714}
    val_dict = ca.get_betweeness_cent(G, **kwargs)
    print(val_dict)
    for node, value in val_dict.items():
        assert_almost_equal(value, ref_values[node])

def test_get_closeness_cent(G, kwargs):
    ref_values = {'1': 0.2962962962962963, '2': 0.36363636363636365, 
                  '4': 0.47058823529411764, '3': 0.36363636363636365, 
                  '5': 0.5, '6': 0.47058823529411764, 
                  '7': 0.36363636363636365, '8': 0.2962962962962963, 
                  '9': 0.36363636363636365}
    val_dict = ca.get_closeness_cent(G, **kwargs)
    print(val_dict)
    for node, value in val_dict.items():
        assert_almost_equal(value, ref_values[node])

def test_get_eigenvector_centt(G, kwargs):
    ref_values = {'1': 0.2628655560595658, '2': 0.300750477503772, 
                  '4': 0.4253254041760193, '3': 0.30075047750377204, 
                  '5': 0.3717480344601847, '6': 0.4253254041760208, 
                  '7': 0.30075047750377376, '8': 0.2628655560595674, 
                  '9': 0.30075047750377376}
    val_dict = ca.get_eigenvector_cent(G, **kwargs)
    print(val_dict)
    for node, value in val_dict.items():
        assert_almost_equal(value, ref_values[node])

def test_get_current_flow_betweenness_cent(G, kwargs):
    ref_values = {'1': 0.125, '2': 0.16964285714285715, '4': 0.5714285714285714, 
                  '3': 0.16964285714285715, '5': 0.5714285714285714, '6': 0.5714285714285715, 
                  '7': 0.16964285714285715, '8': 0.12500000000000025, '9': 0.169642857142857}
    val_dict = ca.get_current_flow_betweenness_cent(G, **kwargs)
    print(val_dict)
    for node, value in val_dict.items():
        assert_almost_equal(value, ref_values[node])

def test_get_current_flow_closeness_cent(G, kwargs):
    ref_values = {'1': 0.05263157894736844, '2': 0.05633802816901411, 
                  '4': 0.07142857142857145, '3': 0.05633802816901411, 
                  '5': 0.07692307692307693, '6': 0.07142857142857148, 
                  '7': 0.05633802816901411, '8': 0.05263157894736844, 
                  '9': 0.05633802816901407}
    val_dict = ca.get_current_flow_closeness_cent(G, **kwargs)
    print(val_dict)
    for node, value in val_dict.items():
        assert_almost_equal(value, ref_values[node])

def test_get_edge_betweenness_cent(G, kwargs):
    ref_values = {'1,2': 0.125, '1,3': 0.125, '2,4': 0.2638888888888889, 
                  '3,4': 0.2638888888888889, '4,5': 0.5555555555555556, 
                  '5,6': 0.5555555555555556, '6,7': 0.2638888888888889, 
                  '6,9': 0.2638888888888889, '7,8': 0.125, '8,9': 0.125}
    val_dict = ca.get_edge_betweenness_cent(G, **kwargs)
    print(val_dict)
    for node, value in val_dict.items():
        assert_almost_equal(value, ref_values[node])

def get_edge_current_flow_betweenness_cent(G, kwargs):
    ref_values = {'1,2': 0.125, '1,3': 0.125, '2,4': 0.2638888888888889, 
                  '3,4': 0.2638888888888889, '4,5': 0.5555555555555556, 
                  '5,6': 0.5555555555555556, '6,7': 0.2638888888888889, 
                  '6,9': 0.2638888888888889, '7,8': 0.125, '8,9': 0.125}
    val_dict = ca.get_edge_current_flow_betweenness_cent(G, **kwargs)
    print(val_dict)
    for node, value in val_dict.items():
        assert_almost_equal(value, ref_values[node])

def test_get_centrality_dict(centrality_dict, ref_name):
    node_dict, edge_dict = centrality_dict
    # check nodes
    node_df = pd.read_csv(ref_name['node'], sep = '\t')
    for measure, val_dict in node_dict.items():
        values = [d for n, d in val_dict.items()]
        ref_values = list(node_df[measure])
        if measure == "node" or measure == "name":
            assert values == ref_values
        else:
            assert_almost_equal(values, ref_values)
    # check edges
    edge_df = pd.read_csv(ref_name['edge'], sep = '\t')
    for measure, val_dict in edge_dict.items():
        values = [d for n, d in val_dict.items()]
        ref_values = list(edge_df[measure])
        if measure == "edge":
            assert values == ref_values
        else:
            assert_almost_equal(values, ref_values)

