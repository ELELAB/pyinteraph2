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
    return os.path.join(request.fspath.dirname, 'data/two_chains')

@pytest.fixture
def data_files(ref_dir):
    return { 
             'pdb' : os.path.join(ref_dir, 'sim.prot.twochains.pdb'),
             'psn' : os.path.join(ref_dir, 'sc-graph_twochains_all.dat')
           }

@pytest.fixture
def ref_name(ref_dir):
    return {
             'all_csv' : os.path.join(ref_dir, 'all_paths_3.txt'),
             'all_dat' : os.path.join(ref_dir, 'all_paths_3.dat'),
             'metapath' : os.path.join(ref_dir, 'metapath.dat'),
             'metapath_csv' : os.path.join(ref_dir, 'metapath.csv')
           }

@pytest.fixture
def data(data_files):
    return pa.build_graph(data_files['psn'], data_files['pdb'])

@pytest.fixture
def metapath_edge():
    return [('A', 'B', 7/32), ('A', 'C', 7/32), ('B', 'D', 12/32), 
            ('C', 'D', 12/32), ('D', 'E', 24/32), ('E', 'F', 12/32), 
            ('E', 'G', 12/32), ('F', 'H', 7/32), ('G', 'H', 7/32)]

@pytest.fixture
def metapath_node():
    return {'A' : 13/32, 'B' : 13/32, 'C' : 13/32, 'D' : 27/32,
            'E' : 27/32, 'F' : 13/32, 'G' : 13/32, 'H' : 13/32}

@pytest.fixture
def example_metapath(metapath_edge, metapath_node):
    G = nx.Graph()
    G.add_weighted_edges_from(metapath_edge)
    for n in G.nodes:
        G.add_node(n, n_weight=metapath_node[n])
    return G

@pytest.fixture
def id():
    return ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']

# Shortest path fixtures
@pytest.fixture
def source(data):
    return pa.convert_input_to_list(user_input = "A1:A2,A57",
                                    identifiers = data[0])

@pytest.fixture
def target(data):
    return pa.convert_input_to_list(user_input = "B1042",
                                    identifiers = data[0])

@pytest.fixture
def shortest_path(example_metapath):
    return pa.get_shortest_paths(graph = example_metapath,
                                 source = 'A',
                                 target = 'H')

@pytest.fixture
def all_path(data, source, target):
    return pa.get_all_simple_paths(graph = data[2],
                                   source = source,
                                   target = target,
                                   maxl = 3)

@pytest.fixture
def all_path_graph(data, all_path):
    return pa.get_persistence_graph(graph = data[2], 
                                    paths = all_path, 
                                    identifiers = data[0])

@pytest.fixture
def all_table(data, all_path):
    return pa.sort_paths(graph = data[2],
                         paths = all_path,
                         sort_by = "path")

# Metapath fixtures
@pytest.fixture
def combinations(data):
    return pa.get_combinations(res_id = data[0],
                               res_space = 3)

@pytest.fixture
def all_shortest_paths(example_metapath, id):
    return pa.get_all_shortest_paths(graph = example_metapath, 
                                     res_id = id, 
                                     res_space = 1)

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
def metapath(example_metapath, id):
    return pa.get_metapath(graph = example_metapath,
                                      res_id = id,
                                      res_space = 1,
                                      node_threshold = 0.3,
                                      edge_threshold = 0.3,
                                      normalize = False)

@pytest.fixture
def reordered_graph(data):
    graph = pa.get_metapath(graph = data[2],
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

# Test functions required for shortest/simple paths
def test_convert_input_to_list(source):
    source.sort()
    ref = ['A1', 'A2', 'A57']
    assert source == ref

def test_get_shortest_paths(example_metapath):
    ref_paths = [['A', 'B', 'D', 'E', 'F', 'H'], ['A', 'C', 'D', 'E', 'F', 'H'], 
                 ['A', 'B', 'D', 'E', 'G', 'H'], ['A', 'C', 'D', 'E', 'G', 'H']]
    paths = pa.get_shortest_paths(example_metapath, 'A', 'H')
    assert paths == ref_paths

def test_get_all_simple_paths(example_metapath):
    ref_paths = [['A', 'B', 'D', 'E', 'F'], ['A', 'C', 'D', 'E', 'F']]
    paths = pa.get_all_simple_paths(example_metapath, 'A', 'F', 5)
    assert paths == ref_paths

def test_get_persistence_graph(example_metapath, shortest_path, id):
    # Using the shortest paths from the example metapath (with source A and sink H)
    # should result in a graph that is identical to the example metapath
    graph = pa.get_persistence_graph(example_metapath, shortest_path, id)
    assert nx.is_isomorphic(example_metapath, graph)

def test_sort_paths(example_metapath, shortest_path):
    ref_path = [['A', 'B', 'D', 'E', 'F', 'H'], ['A', 'B', 'D', 'E', 'G', 'H'],
                ['A', 'C', 'D', 'E', 'F', 'H'], ['A', 'C', 'D', 'E', 'G', 'H']]
    path_table = pa.sort_paths(example_metapath, shortest_path, "path")
    for i in range(len(ref_path)):
        assert path_table[i][0] == ref_path[i]
        assert path_table[i][1] == ref_path[i][0]
        assert path_table[i][2] == ref_path[i][-1]
        assert path_table[i][3] == len(ref_path[i])
        weights = [example_metapath[ref_path[i][j]][ref_path[i][j+1]]["weight"] \
                    for j in range(len(ref_path[i]) - 1)]
        cumulative = sum(weights)
        average = cumulative/len(weights)
        assert path_table[i][4] == cumulative
        assert path_table[i][5] == average

# Check output files
def test_sort_paths_all_table(all_table, ref_name):
    ref_csv = []
    # Recreate out
    with open(ref_name['all_csv'], "r") as f:
        for line in f:
            # remove white space and split line
            li, s, t, l, w1, w2 = line.rstrip().split('\t')
            # change to correct format
            line = (li.split(','), s, t, int(l), float(w1), float(w2))
            ref_csv.append(line)
    assert all_table == ref_csv

def test_all_path_graph(all_path_graph, ref_name):
    ref_graph = np.loadtxt(ref_name['all_dat'])
    graph = nx.to_numpy_matrix(all_path_graph)
    assert_equal(graph, ref_graph)

# Test functions required for metapath calculation
def test_get_combinations(data):
    # Check if the combination distances are correct
    combinations = pa.get_combinations(data[0], 3)
    for combination in combinations:
        idx1 = data[0].index(combination[0])
        idx2 = data[0].index(combination[1])
        if combination[0][0] == combination[1][0]:
            assert abs(idx1 - idx2) >= 3

def test_get_all_shortest_paths(example_metapath, id):
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
    paths = pa.get_all_shortest_paths(example_metapath, id, 1)
    assert paths == ref_paths

def test_graph_from_paths(example_metapath, all_shortest_paths):
    metapath = pa.get_graph_from_paths(all_shortest_paths)
    assert nx.is_isomorphic(metapath, example_metapath)
    for u, v, d in metapath.edges(data = True):
        assert d['e_weight'] == example_metapath[u][v]['weight']
    for n, d in metapath.nodes(data = True):
        assert d['n_weight'] == example_metapath.nodes()[n]["n_weight"]

def test_filter_graph(filtered_graph):
    # Check that values are within the filtered range
    for _, _, d in filtered_graph.edges(data = True):
        assert d['e_weight'] > 0.3
    for _, d in filtered_graph.nodes(data = True):
        assert d['n_weight'] > 0.3

def test_normalized_graph(graph_from_paths, normalized_graph):
    # Check that original values can be recalculated from normalized values
    max_edge = max([d['e_weight'] for u, v, d in graph_from_paths.edges(data = True)])
    max_node = max([d['n_weight'] for n, d in graph_from_paths.nodes(data = True)])
    for u, v, d in normalized_graph.edges(data = True):
        assert_almost_equal(graph_from_paths[u][v]['e_weight']*100, d['e_weight']*max_edge)
    for n, d in normalized_graph.nodes(data = True):
        assert_almost_equal(graph_from_paths.nodes()[n]['n_weight'], d['n_weight']*max_node)

def test_metapath(metapath, example_metapath):
    # Remove these edges since a filter of 0.3 was used
    edges = [('A', 'B'), ('A', 'C'), ('F', 'H'), ('G', 'H')]
    example_metapath.remove_edges_from(edges)
    for u, v, d in metapath.edges(data = True):
        assert d['e_weight'] == example_metapath[u][v]['weight']
    for n, d in metapath.nodes(data = True):
        assert d['n_weight'] == example_metapath.nodes()[n]["n_weight"]

def test_reorder_graph(reordered_graph, data):
    nodes = list(reordered_graph.nodes())
    for i in range(len(data[0])):
        assert nodes[i] == data[0][i]

# Test output file
def test_metapath_matrix(metapath_matrix, ref_name):
    ref_metapath = np.loadtxt(ref_name['metapath'])
    assert_equal(metapath_matrix, ref_metapath)

def test_metapath_csv(data, reordered_graph, ref_name):
    with open(ref_name['metapath_csv']) as fh:
        ref_csv = fh.read()

    identifiers, residue_names, _ = data
    table = pa.generate_metapath_table(reordered_graph,
                                       identifiers,
                                       residue_names,
                                       normalize=False)

    assert ref_csv == table
