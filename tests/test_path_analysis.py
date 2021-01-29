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
             'sb' : os.path.join(ref_dir, 'sb-graph_twochains_all.dat'),
             'hc' : os.path.join(ref_dir, 'hc-graph.dat'),
             'hb' : os.path.join(ref_dir, 'hb-graph.dat'),
           }

@pytest.fixture
def sb_graph(data_files):
    return pa.build_graph(data_files['sb'], data_files['pdb'])

@pytest.fixture
def sb_source(sb_graph):
    return pa.convert_input_to_list(user_input = "A4:A6,A57,A58,A10",
                                    identifiers = sb_graph[0])

@pytest.fixture
def sb_target(sb_graph):
    return pa.convert_input_to_list(user_input = "A5,A11,B1042",
                                    identifiers = sb_graph[0])

@pytest.fixture
def sb_shortest_path(sb_graph, sb_source, sb_target):
    return pa.get_shortest_paths(graph = sb_graph[1],
                                 source = sb_source,
                                 target = sb_target)

@pytest.fixture
def sb_all_path(sb_graph, sb_source, sb_target):
    return pa.get_all_simple_paths(graph = sb_graph[1],
                                 source = sb_source,
                                 target = sb_target,
                                 maxl = 3)

@pytest.fixture
def sb_shortest_path_graph(sb_graph, sb_shortest_path):
    return pa.get_persistence_graph(graph = sb_graph[1], 
                                    paths = sb_shortest_path, 
                                    identifiers = sb_graph[0])

@pytest.fixture
def sb_all_path_graph(sb_graph, sb_all_path):
    return pa.get_persistence_graph(graph = sb_graph[1], 
                                    paths = sb_all_path, 
                                    identifiers = sb_graph[0])

@pytest.fixture
def sb_shortest_table(sb_graph, sb_shortest_path):
    return pa.sort_paths(graph = sb_graph[1],
                         paths = sb_shortest_path,
                         sort_by = "average_weight")

@pytest.fixture
def sb_all_table(sb_graph, sb_all_path):
    return pa.sort_paths(graph = sb_graph[1],
                         paths = sb_all_path,
                         sort_by = "average_weight")

@pytest.fixture
def sb_ref_name(ref_dir):
    return {
             'shortest_csv' : os.path.join(ref_dir, 'shortest_paths.txt'),
             'shortest_dat' : os.path.join(ref_dir, 'shortest_paths.dat'),
             'all_csv' : os.path.join(ref_dir, 'all_paths_3.txt'),
             'all_dat' : os.path.join(ref_dir, 'all_paths_3.dat')
           }


# Test shortest paths
def test_shortest_path(sb_shortest_table, sb_ref_name):
    ref_csv = []
    with open(sb_ref_name['shortest_csv'], "r") as f:
        for line in f:
            # remove white space and split line
            print(line)
            li, s, t, l, w1, w2 = line.rstrip().split('\t')
            # change to correct format
            line = (eval(li), s, t, int(l), float(w1), float(w2))
            ref_csv.append(line)
    print(sb_shortest_table)
    assert sb_shortest_table == ref_csv

def test_shortest_path_graph(sb_shortest_path_graph, sb_ref_name):
    ref_graph = np.loadtxt(sb_ref_name['shortest_dat'])
    graph = nx.to_numpy_matrix(sb_shortest_path_graph)
    assert_equal(graph, ref_graph)

# Test simple paths
def test_all_path(sb_all_table, sb_ref_name):
    ref_csv = []
    with open(sb_ref_name['all_csv'], "r") as f:
        for line in f:
            # remove white space and split line
            print(line)
            li, s, t, l, w1, w2 = line.rstrip().split('\t')
            # change to correct format
            line = (eval(li), s, t, int(l), float(w1), float(w2))
            ref_csv.append(line)
    print(sb_all_table)
    assert sb_all_table == ref_csv

def test_all_path_graph(sb_all_path_graph, sb_ref_name):
    ref_graph = np.loadtxt(sb_ref_name['all_dat'])
    graph = nx.to_numpy_matrix(sb_all_path_graph)
    assert_equal(graph, ref_graph)



# Test metapath

