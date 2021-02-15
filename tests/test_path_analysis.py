import os
import os.path
import numpy as np
import networkx as nx
import pytest
from numpy.testing import assert_almost_equal, assert_equal
from pyinteraph import path_analysis as pa

# Load files
@pytest.fixture
def ref_dir(request):
    return os.path.join(request.fspath.dirname, '../examples')

@pytest.fixture
def data_files(ref_dir):
    return { 
             'pdb' : os.path.join(ref_dir, 'sim.prot.twochains.pdb'),
             'psn' : os.path.join(ref_dir, 'sc-graph_twochains_all.dat')
           }

@pytest.fixture
def ref_name(ref_dir):
    return {
             'shortest_csv' : os.path.join(ref_dir, 'shortest_paths.txt'),
             'shortest_dat' : os.path.join(ref_dir, 'shortest_paths.dat'),
             'all_csv' : os.path.join(ref_dir, 'all_paths_3.txt'),
             'all_dat' : os.path.join(ref_dir, 'all_paths_3.dat'),
             'metapath' : os.path.join(ref_dir, 'metapath.dat'),
             'metapath_norm' : os.path.join(ref_dir, 'metapath_norm.dat')
           }

@pytest.fixture
def data(data_files):
    return pa.build_graph(data_files['psn'], data_files['pdb'])

@pytest.fixture
def source(data):
    return pa.convert_input_to_list(user_input = "A1:A2,A57",
                                    identifiers = data[0])

@pytest.fixture
def target(data):
    return pa.convert_input_to_list(user_input = "B1042",
                                    identifiers = data[0])

@pytest.fixture
def shortest_path(data, source, target):
    return pa.get_shortest_paths(graph = data[1],
                                 source = source,
                                 target = target)

@pytest.fixture
def all_path(data, source, target):
    return pa.get_all_simple_paths(graph = data[1],
                                   source = source,
                                   target = target,
                                   maxl = 3)

@pytest.fixture
def shortest_path_graph(data, shortest_path):
    return pa.get_persistence_graph(graph = data[1], 
                                    paths = shortest_path, 
                                    identifiers = data[0])

@pytest.fixture
def all_path_graph(data, all_path):
    return pa.get_persistence_graph(graph = data[1], 
                                    paths = all_path, 
                                    identifiers = data[0])

@pytest.fixture
def shortest_table(data, shortest_path):
    return pa.sort_paths(graph = data[1],
                         paths = shortest_path,
                         sort_by = "path")

@pytest.fixture
def all_table(data, all_path):
    return pa.sort_paths(graph = data[1],
                         paths = all_path,
                         sort_by = "path")

@pytest.fixture
def combinations(data):
    return pa.get_combinations(res_id = data[0],
                               res_space = 3)

@pytest.fixture
def example_metapath():
    edges = [('A', 'B', 7/32), ('A', 'C', 7/32), ('B', 'D', 12/32), 
             ('C', 'D', 12/32), ('D', 'E', 24/32), ('E', 'F', 12/32), 
             ('E', 'G', 12/32), ('F', 'H', 7/32), ('G', 'H', 7/32)]
    nodes = {'A' : 13/32, 'B' : 13/32, 'C' : 13/32, 'D' : 27/32,
             'E' : 27/32, 'F' : 13/32, 'G' : 13/32, 'H' : 13/32}
    G = nx.Graph()
    G.add_weighted_edges_from(edges)
    for n in G.nodes:
        G.add_node(n, n_weight=nodes[n])
    return G

@pytest.fixture
def all_shortest_paths(example_metapath):
    id = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
    return pa.get_all_shortest_paths(graph = example_metapath, 
                                     res_id = id, 
                                     res_space = 1)

# @pytest.fixture
# def all_shortest_paths(data):
#     return pa.get_all_shortest_paths(graph = data[1],
#                                      res_id = data[0],
#                                      res_space = 0)

@pytest.fixture
def graph_from_paths(all_shortest_paths):
    return pa.get_graph_from_paths(all_shortest_paths)

@pytest.fixture
def filtered_graph(graph_from_paths):
    return pa.filter_graph(graph_from_paths, 0.3, 0.3)

@pytest.fixture
def normalized_graph(graph_from_paths):
    return pa.normalize_graph(graph_from_paths)

@pytest.fixture
def metapath(example_metapath):
    id = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
    return pa.get_metapath(graph = example_metapath,
                                      res_id = id,
                                      res_space = 1,
                                      node_threshold = 0.3,
                                      edge_threshold = 0.3,
                                      normalize = False)

@pytest.fixture
def reordered_graph(data):
    graph = pa.get_metapath(graph = data[1],
                               res_id = data[0],
                               res_space = 3,
                               node_threshold = 0.1,
                               edge_threshold = 0.1,
                               normalize = False)
    graph = pa.reorder_graph(graph, data[0])
    return graph

@pytest.fixture
def metapath_matrix(reordered_graph):
    matrix = nx.to_numpy_matrix(reordered_graph)
    return matrix

# @pytest.fixture
# def metapath_norm(data):
#     metapath = pa.get_metapath(graph = data[1],
#                                res_id = data[0],
#                                res_space = 3,
#                                node_threshold = 0.1,
#                                edge_threshold = 0.1,
#                                normalize = True)
#     metapath = pa.reorder_graph(metapath, data[0])
#     metapath = nx.to_numpy_matrix(metapath)
#     return metapath


# # Test path helper functions
# def test_convert_input_to_list(source):
#     source.sort()
#     ref = ['A1', 'A2', 'A57']
#     assert source == ref

# # Test shortest path outputs
# def test_sort_paths_shortest_table(shortest_table, ref_name):
#     ref_csv = []
#     with open(ref_name['shortest_csv'], "r") as f:
#         for line in f:
#             # remove white space and split line
#             li, s, t, l, w1, w2 = line.rstrip().split('\t')
#             # change to correct format
#             line = (li.split(','), s, t, int(l), float(w1), float(w2))
#             ref_csv.append(line)
#     assert shortest_table == ref_csv

# def test_shortest_path_graph(shortest_path_graph, ref_name):
#     ref_graph = np.loadtxt(ref_name['shortest_dat'])
#     graph = nx.to_numpy_matrix(shortest_path_graph)
#     assert_equal(graph, ref_graph)

# # Test simple paths
# def test_sort_paths_all_table(all_table, ref_name):
#     ref_csv = []
#     with open(ref_name['all_csv'], "r") as f:
#         for line in f:
#             # remove white space and split line
#             li, s, t, l, w1, w2 = line.rstrip().split('\t')
#             # change to correct format
#             line = (li.split(','), s, t, int(l), float(w1), float(w2))
#             ref_csv.append(line)
#     assert all_table == ref_csv

# def test_all_path_graph(all_path_graph, ref_name):
#     ref_graph = np.loadtxt(ref_name['all_dat'])
#     graph = nx.to_numpy_matrix(all_path_graph)
#     assert_equal(graph, ref_graph)

# Test metapath
def test_get_combinations(data):
    combinations = pa.get_combinations(data[0], 3)
    for combination in combinations:
        idx1 = data[0].index(combination[0])
        idx2 = data[0].index(combination[1])
        if combination[0][0] == combination[1][0]:
            assert abs(idx1 - idx2) >= 3

def test_get_all_shortest_paths(example_metapath):
    ref_paths = [['A', 'B', 'D'], ['A', 'C', 'D'], ['A', 'B', 'D', 'E'], 
                 ['A', 'C', 'D', 'E'], ['A', 'B', 'D', 'E', 'F'], 
                 ['A', 'C', 'D', 'E', 'F'], ['A', 'B', 'D', 'E', 'G'], 
                 ['A', 'C', 'D', 'E', 'G'], ['A', 'B', 'D', 'E', 'F', 'H'], 
                 ['A', 'C', 'D', 'E', 'F', 'H'], ['A', 'B', 'D', 'E', 'G', 'H'], 
                 ['A', 'C', 'D', 'E', 'G', 'H'], ['B', 'A', 'C'], ['B', 'D', 'C'], 
                 ['B', 'D', 'E'], ['B', 'D', 'E', 'F'], ['B', 'D', 'E', 'G'], 
                 ['B', 'D', 'E', 'F', 'H'], ['B', 'D', 'E', 'G', 'H'], 
                 ['C', 'D', 'E'], ['C', 'D', 'E', 'F'], ['C', 'D', 'E', 'G'], 
                 ['C', 'D', 'E', 'F', 'H'], ['C', 'D', 'E', 'G', 'H'], 
                 ['D', 'E', 'F'], ['D', 'E', 'G'], ['D', 'E', 'F', 'H'], 
                 ['D', 'E', 'G', 'H'], ['E', 'F', 'H'], ['E', 'G', 'H'], 
                 ['F', 'E', 'G'], ['F', 'H', 'G']]
    id = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
    paths = pa.get_all_shortest_paths(example_metapath, id, 1)
    assert paths == ref_paths

def test_graph_from_paths(example_metapath, all_shortest_paths):
    metapath = pa.get_graph_from_paths(all_shortest_paths)
    assert nx.is_isomorphic(metapath, example_metapath)
    for u, v, d in metapath.edges(data= True):
        assert d['e_weight'] == example_metapath[u][v]['weight']
    for n, d in metapath.nodes(data= True):
        assert d['n_weight'] == example_metapath.nodes()[n]["n_weight"]

def test_filter_graph(filtered_graph):
    for _, _, d in filtered_graph.edges(data = True):
        assert d['e_weight'] > 0.3
    for _, d in filtered_graph.nodes(data = True):
        assert d['n_weight'] > 0.3

def test_normalized_graph(graph_from_paths, normalized_graph):
    max_edge = max([d['e_weight'] for u, v, d in graph_from_paths.edges(data = True)])
    max_node = max([d['n_weight'] for n, d in graph_from_paths.nodes(data = True)])
    for u, v, d in normalized_graph.edges(data = True):
        assert_almost_equal(graph_from_paths[u][v]['e_weight'], d['e_weight']*max_edge)
    for n, d in normalized_graph.nodes(data = True):
        assert_almost_equal(graph_from_paths.nodes()[n]['n_weight'], d['n_weight']*max_node)

def test_metapath(metapath, example_metapath):
    edges = [('A', 'B'), ('A', 'C'), ('F', 'H'), ('G', 'H')]
    example_metapath.remove_edges_from(edges)
    for u, v, d in metapath.edges(data= True):
        assert d['e_weight'] == example_metapath[u][v]['weight']
    for n, d in metapath.nodes(data= True):
        assert d['n_weight'] == example_metapath.nodes()[n]["n_weight"]

def test_reorder_graph(reordered_graph, data):
    nodes = list(reordered_graph.nodes())
    for i in range(len(data[0])):
        assert nodes[i] == data[0][i]

def test_metapath_matrix(metapath_matrix, ref_name):
    ref_metapath = np.loadtxt(ref_name['metapath'])
    assert_equal(metapath_matrix, ref_metapath)

