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
def data(data_files):
    return pa.build_graph(data_files['psn'], data_files['pdb'])

@pytest.fixture
def source(data):
    return pa.convert_input_to_list(user_input = "A2,A57",
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
                         sort_by = "average_weight")

@pytest.fixture
def all_table(data, all_path):
    return pa.sort_paths(graph = data[1],
                         paths = all_path,
                         sort_by = "average_weight")
                        
@pytest.fixture
def metapath(data):
    metapath = pa.get_metapath(graph = data[1],
                               res_id = data[0],
                               res_space = 3,
                               node_threshold = 0.1,
                               edge_threshold = 0.1,
                               normalize = False)
    metapath = pa.reorder_graph(metapath, data[0])
    metapath = nx.to_numpy_matrix(metapath)
    return metapath

@pytest.fixture
def metapath_norm(data):
    metapath = pa.get_metapath(graph = data[1],
                               res_id = data[0],
                               res_space = 3,
                               node_threshold = 0.1,
                               edge_threshold = 0.1,
                               normalize = True)
    metapath = pa.reorder_graph(metapath, data[0])
    metapath = nx.to_numpy_matrix(metapath)
    return metapath


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


# Test shortest paths
def test_shortest_path(shortest_table, ref_name):
    ref_csv = []
    with open(ref_name['shortest_csv'], "r") as f:
        for line in f:
            # remove white space and split line
            li, s, t, l, w1, w2 = line.rstrip().split('\t')
            # change to correct format
            line = (eval(li), s, t, int(l), float(w1), float(w2))
            ref_csv.append(line)
    assert shortest_table == ref_csv

def test_shortest_path_graph(shortest_path_graph, ref_name):
    ref_graph = np.loadtxt(ref_name['shortest_dat'])
    graph = nx.to_numpy_matrix(shortest_path_graph)
    assert_equal(graph, ref_graph)

# Test simple paths
def test_all_path(all_table, ref_name):
    ref_csv = []
    with open(ref_name['all_csv'], "r") as f:
        for line in f:
            # remove white space and split line
            li, s, t, l, w1, w2 = line.rstrip().split('\t')
            # change to correct format
            line = (eval(li), s, t, int(l), float(w1), float(w2))
            ref_csv.append(line)
    assert all_table == ref_csv

def test_all_path_graph(all_path_graph, ref_name):
    ref_graph = np.loadtxt(ref_name['all_dat'])
    graph = nx.to_numpy_matrix(all_path_graph)
    assert_equal(graph, ref_graph)

# Test metapath
def test_metapath(metapath, ref_name):
    ref_metapath = np.loadtxt(ref_name['metapath'])
    assert_equal(metapath, ref_metapath)

def test_metapath_norm(metapath_norm, ref_name):
    ref_metapath = np.loadtxt(ref_name['metapath_norm'])
    print((ref_metapath == metapath_norm).sum())
    assert_almost_equal(metapath_norm, ref_metapath)

