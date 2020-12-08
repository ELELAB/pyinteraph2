import argparse
import logging as log
import numpy as np
import networkx as nx
import MDAnalysis as mda
import itertools
import re

def build_graph(fname, pdb = None):
    """Build a graph from the provided matrix"""

    try:
        data = np.loadtxt(fname)
    except:
        errstr = f"Could not load file {fname} or wrong file format."
        raise ValueError(errstr)
    # if the user provided a reference structure
    if pdb is not None:
        try:
            # generate a Universe object from the PDB file
            u = mda.Universe(pdb)
        except Exception as e:
            errstr = \
                f"Exception caught during creation of the Universe: {e}"
            raise ValueError(errstr)      
        # generate identifiers for the nodes of the graph
        identifiers = [f"{r.segment.segid}{r.resnum}" for r in u.residues]
    # if the user did not provide a reference structure
    else:
        # generate automatic identifiers going from 1 to the
        # total number of residues considered
        identifiers = [str(i) for i in range(1, data.shape[0]+1)]
    
    # generate a graph from the data loaded
    G = nx.Graph(data)
    # set the names of the graph nodes (in place)
    node_names = dict(zip(range(data.shape[0]), identifiers))
    nx.relabel_nodes(G, mapping = node_names, copy = False)
    # return the idenfiers and the graph
    return identifiers, G

def convert_input_to_list(user_input, identifiers, pdb = False):
    """Take in a string (e.g. A12:A22,A13... if a PDB file is supplied)
    and a list of names of all the residues (graph nodes). Replaces the 
    range indicated by the colon with all resiues in that range and 
    keeps all residues separated by commas. Removes duplicates. Takes in
    a string e.g. 1,3,4:56 if no PDB file is supplied.
    """

    # Check if PDB file is supplied
    if pdb:
        # Find all residues separated by commas by
        # replacing all colon residues with ''
        input_comma = re.sub('\w+:\w+', '', user_input)
        # Find all residues separated by colons
        input_colon = re.findall('\w+:\w+', user_input)
    else:
        # No PDB file present
        input_comma = re.sub('\d+:\d+', '', user_input)
        input_colon = re.findall('\d+:\d+', user_input)
    comma_list = input_comma.split(',')
    # Remove empty residues
    comma_list = [res for res in comma_list if res != '']
    # Report if any residues are not in the PDB
    try:
        for res in comma_list:
            identifiers.index(res)
    except Exception:
        raise ValueError(f"Residue not in PDB or incorrect format: {res}")
    colon_replace = []
    # Substitute range of residues with the actual residues
    for inp in input_colon:
        try:
            # Create list of size two with start and end of range
            colon_split = inp.split(':')
            # Find the index of those res in the indentifiers list
            index = [identifiers.index(res) for res in colon_split]
            # Replace with the residues in that range
            inp_replace = identifiers[index[0]:index[1]+1]
            # Concatenate to list
            colon_replace += inp_replace
        except Exception:
            # Report if the specified range does not exist in the PDB
            raise ValueError(f"Residue range not in PDB or incorrect format: {inp}")
    # Add both lists
    input_list = comma_list + colon_replace
    # Remove duplicates
    input_list = list(set(input_list))
    return input_list

def get_shortest_paths(graph, source, target, maxl):
    """Find all shortest paths between all combinations of source and
    target.
    """
    
    # Get all combinations
    combinations = itertools.product(source, target)
    # Get all shortest paths
    paths = []
    for node1, node2 in combinations:
        if node1 != node2:
            try:
                # Get a list of shortest paths and append to output
                path = list(nx.algorithms.shortest_paths.generic.all_shortest_paths(\
                                G = graph, \
                                source = node1, \
                                target = node2))
                for p in path:
                    # Check that path is not longer than the maximum allowed length
                    if len(p) <= maxl:
                        paths.append(p)
            except nx.NetworkXNoPath:
                # If no path is found log info
                log.warning(f"No path found between {node1} and {node2}")
    return paths

def get_all_simple_paths(graph, source, target, maxl):
    """Find all simple paths between all combinations of source and
    target.
    """

    # Get all combinations
    combinations = itertools.product(source, target)
    # Get all simple paths
    paths = []
    for node1, node2 in combinations:
        # Get all simple paths for each combination of source and target
        path = list(nx.algorithms.simple_paths.all_simple_paths(\
                        G = graph, \
                        source = node1, \
                        target = node2, \
                        cutoff= maxl))
        # Only add paths to output if they exist
        for p in path:
            if len(p) > 0:
                paths.append(p)
    return paths

def sort_paths(graph, paths, sort_by):
    """Takes in a list of paths and sorts them."""
    
    # Get source and target
    source = [p[0] for p in paths]
    target = [p[-1] for p in paths]
    # Calculate length of path
    lengths = [len(p) for p in paths]
    # Calculate weights of path
    weights = \
              [[graph[p[i]][p[i+1]]["weight"] for i in range(len(p)-1)] \
                  for p in paths]
    sum_weights = [np.sum(w) for w in weights]
    avg_weights = [np.mean(w) for w in weights]
    # Sort paths
    paths = zip(paths, source, target, lengths, sum_weights, avg_weights)
    if sort_by == "length":
        key = lambda x: x[3]
        reverse = False
    elif sort_by == "cumulative_weight":
        key = lambda x: x[4]
        reverse = True
    elif sort_by == "average_weight":
        key = lambda x: x[5]
        reverse = True
    sorted_paths = sorted(paths, key = key, reverse = reverse)
    return sorted_paths

def get_combinations(res_id, res_space):
    """ Takes in a list of residue identifiers and returns all pairs of
    residues that are at least res_space apart if they are on the same
    chain.
    """
    
    # Get all index combinations
    combinations = itertools.combinations(range(len(res_id)), 2)
    # Get all residue combinations if they are res_space distance apart
    # Or they are on different chains
    combinations = [(res_id[idx1], res_id[idx2]) for idx1, idx2 in combinations \
                    if abs(idx1 - idx2) > res_space \
                    or res_id[idx1][0] != res_id[idx2][0]]
    return combinations


def get_all_shortest_paths(graph, res_id, res_space):
    """Find all shortest paths between all combinations of nodes in the
    graph that are at least res_space distance apart.
    """
    
    # Get all combinations
    combinations = get_combinations(res_id, res_space)
    # Get all shortest paths
    paths = []
    for node1, node2 in combinations:
        try:
            # Get a list of shortest paths and append to output
            path = list(nx.algorithms.shortest_paths.generic.all_shortest_paths(\
                            G = graph, \
                            source = node1, \
                            target = node2))
            for p in path:
                # Check that path is not longer than the maximum allowed length
                paths.append(p)
        except Exception:
            # If no path is found log info
            log.debug(f"No path found between {node1} and {node2}")
    return paths

def get_common_nodes(paths, threshold):
    """Takes in an list of paths and returns the nodes which are more 
    common than the provided threshold.
    """
    
    # Get length of longest path
    maxl = max([len(p) for p in paths])
    # Convert path list to array where all paths are equal sized
    # Use the maximum path length as an upper bound
    paths_array = np.array([p + [0]*(maxl-len(p)) for p in paths])
    # Get unique nodes
    unique_nodes = np.unique(paths_array)
    unique_nodes = unique_nodes[unique_nodes != "0"]
    # Get node percentage
    node_count = np.array([(node == paths_array).sum() for node in unique_nodes])
    node_perc = node_count/node_count.sum()
    # Select common nodes
    common_nodes = unique_nodes[node_perc > threshold]
    common_nodes = list(common_nodes)
    return common_nodes

def get_graph_from_paths(paths):
    """Takes in a list of paths and returns a corresponding graph where
    the weight of each edge is its count.
    """

    graph = nx.Graph()
    for path in paths:
        for i in range(len(path) - 1):
            # Get edge
            node1 = path[i]
            node2 = path[i+1]
            # Increment weight if edge exists
            if graph.has_edge(node1, node2):
                graph[node1][node2]["weight"] += 1
            # Add edge otherwise
            else:
                graph.add_edge(node1, node2, weight = 1)
    return graph


def get_common_edges(graph, threshold):
    """Takes in a graph where the edge weights are the count of the edge
    and returns a list of edges which are larger than the provided 
    threshold.
    """

    # Initialize arrays of edges and their count
    edges = np.array(graph.edges)
    counts = np.array([graph[e[0]][e[1]]["weight"] for e in edges])
    # Calculate percentage occurance for each path
    perc = counts/counts.sum()
    # Choose paths larger than the threshold
    common_edges = edges[perc > threshold]
    # Convert array to list of tuples
    common_edges = list(map(tuple, common_edges))
    return common_edges

def get_metapath(graph, res_id, res_space, node_threshold, edge_threshold):
    """Takes in a list of paths, an edge threshold and a node threshold
    and returns a list of metapaths where each metapath contains all 
    nodes and edges above their respective thresholds.
    """
    
    # Calculate all shortest paths
    paths = get_all_shortest_paths(graph, res_id, res_space)
    # Create graph from path list
    paths_graph = get_graph_from_paths(paths)

    # Get common nodes
    common_nodes = get_common_nodes(paths, node_threshold)
    # Get common edges
    common_edges = get_common_edges(paths_graph, edge_threshold)
   
    common_paths = []
    if len(common_nodes) == 0:
        # Warn if no common nodes found
        warn_str = f"No Nodes found with a frequency higher than: {node_threshold}"
        log.warning(warn_str)
    elif len(common_edges) == 0:
        # Warn if no common edges are found
        warn_str = f"No Edges found with a frequency higher than: {edge_threshold}"
        log.warning(warn_str)
    # Common edges and nodes exist
    else:
        # Find which paths have the common nodes and edges
        for p in paths:
            edges = [(p[i], p[i+1]) for i in range(len(p) - 1)]
            # Check if required nodes are in path
            for c_node in common_nodes:
                if c_node in p:
                    # Check if required edges are in path
                    for c_edge in common_edges:
                        # Add path if not already added
                        if c_edge in edges and p not in common_paths:
                            common_paths.append(p)
        # Warn if no metapaths found
        if len(common_paths) == 0:
            log.warning("No metapaths found.")
    return common_paths, paths_graph

def write_table(fname, table):
    """Save sorted table as txt file. """
    with open(f"{fname}.txt", "w") as f:
        for p, s, t, l, sum_w, avg_w in table:
            f.write(f"{p}\t{s}\t{t}\t{l}\t{sum_w}\t{avg_w}\n")


def main():

    ######################### ARGUMENT PARSER #########################
    
    description = "Path analysis"
    parser = argparse.ArgumentParser(description= description)

    i_helpstr = ".dat file matrix"
    parser.add_argument("-i", "--input-dat",
                        dest = "input_matrix",
                        help = i_helpstr,
                        type = str)
    
    p_helpstr = "Reference PDB file"
    parser.add_argument("-p", "--pdb",
                        dest = "pdb",
                        help = p_helpstr,
                        default = None,
                        type = str)

    l_default = 10
    l_helpstr = f"Maximum path length (default: {l_default})"
    parser.add_argument("-l", "--maximum-path-length", 
                        dest = "maxl",
                        default = l_default,
                        type = int,
                        help = l_helpstr)

    r_default  = 1
    r_helpstr = f"Residue spacing (default: {r_default})"
    parser.add_argument("-r", "--residue-spacing", 
                        dest = "res_space",
                        default = r_default,
                        type = int,
                        help = r_helpstr)

    e_default  = 0.1
    e_helpstr = f"Edge threshold (default: {e_default})"
    parser.add_argument("-e", "--edge-threshold", 
                        dest = "edge_thresh",
                        default = e_default,
                        type = float,
                        help = e_helpstr)

    n_default  = 0.1
    n_helpstr = f"Node threshold (default: {n_default})"
    parser.add_argument("-n", "--node-threshold", 
                        dest = "node_thresh",
                        default = n_default,
                        type = float,
                        help = n_helpstr)

    a_helpstr = "Calculate all simple paths between " \
                "two residues in the graph"
    parser.add_argument("-a", "--all-paths",
                        dest = "do_paths",
                        action = "store_true",
                        default = False,
                        help = a_helpstr)

    b_choices = ["length", "cumulative_weight", "avg_weight"]
    b_default = "length"
    b_helpstr = "How to sort pathways in output. Possible choices are: " \
                f"{b_choices} (default: {b_default}"
    parser.add_argument("-b", "--sort-paths",
                        dest = "sort_by",
                        choices = b_choices,
                        default = b_default,
                        help =  b_helpstr)

    s_helpstr = "Source residue for paths calculation (see option -p)"
    parser.add_argument("-s", "--source",
                        dest = "source",
                        default = None,
                        type = str,
                        help = s_helpstr)

    t_helpstr = "Target residue for paths calculation (see option -p)"
    parser.add_argument("-t", "--target",
                        dest = "target",
                        default = None,
                        type = str,
                        help = t_helpstr)

    o_default = "paths"
    o_helpstr = "Output file name"
    parser.add_argument("-o", "--output",
                        dest = "output",
                        default = o_default,
                        help = o_helpstr)

    args = parser.parse_args()
    
    # Check user input
    if not args.input_matrix:
        # exit if the adjacency matrix was not speficied
        log.error("Graph adjacency matrix must be specified. Exiting ...")
        exit(1)
    
    # Check if pdb file is present
    if not args.pdb:
        pdb_boolean = False
    else:
        pdb_boolean = True
    
    # Load file, build graphs and get identifiers for graph nodes
    identifiers, graph = build_graph(fname = args.input_matrix, \
                                     pdb = args.pdb)

    # Convert user input to a list of nodes
    source_list = convert_input_to_list(user_input = args.source, \
                                        identifiers = identifiers, \
                                        pdb = pdb_boolean)
    target_list = convert_input_to_list(user_input = args.target, \
                                        identifiers = identifiers, \
                                        pdb = pdb_boolean)
    
    # Choose whether to get shortest paths or all paths
    if args.do_paths:
        all_paths = get_all_simple_paths(graph = graph, \
                              source = source_list, \
                              target = target_list, \
                              maxl = args.maxl)
    else:
        all_paths = get_shortest_paths(graph = graph, \
                                    source = source_list, \
                                    target = target_list, \
                                    maxl = args.maxl)
    
    # Create sorted table from paths
    all_paths_table = sort_paths(graph = graph, \
                                 paths = all_paths, \
                                 sort_by = args.sort_by)
    all_paths_graph = get_graph_from_paths(all_paths)

    # Save table
    write_table(args.output, all_paths_table)

    # Write matrix
    path_matrix = nx.to_numpy_matrix(all_paths_graph)
    np.savetxt(f"{args.output}.dat", path_matrix)
    
    # Get list of metapaths and graph of metapaths
    metapath, metapath_graph = get_metapath(\
                                    graph = graph, \
                                    res_id = identifiers, \
                                    res_space = args.res_space, \
                                    node_threshold = args.node_thresh, \
                                    edge_threshold = args.edge_thresh)

    # Create sorted table from Metapaths
    metapath_table = sort_paths(graph = metapath_graph, \
                                paths = metapath,\
                                sort_by = args.sort_by)

    # Save table
    write_table("metapath", metapath_table)

if __name__ == "__main__":
    main()