import argparse
import logging as log
import numpy as np
import networkx as nx
import itertools

def get_combinations(source, target, res_space):
    """Return an iterator that contains all combinations between source 
    and target that are at least res_space apart. 
    """

    combinations = itertools.combinations(range(source, 
                                                target + 1, 
                                                res_space), 2)
    return combinations

def get_shortest_paths(graph, source, target, maxl, res_space):
    """Find all shortest paths between all combinations of source and
    target.
    """
    
    # Get all combinations
    combinations = get_combinations(source, target, res_space)
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
                if len(p) <= maxl:
                    paths.append(p)
        except nx.NetworkXNoPath:
            pass
            # If no path is found log info
            log.info(f"No path found between {node1} and {node2}")
    return paths

def get_all_paths(graph, source, target, maxl, res_space):
    """Find all simple paths between all combinations of source and
    target.
    """

    # Get all combinations
    combinations = get_combinations(source, target, res_space)
    # Get all simple paths
    paths = []
    for node1, node2 in combinations:
        path = list(nx.algorithms.simple_paths.all_simple_paths(\
                        G = graph, \
                        source = node1, \
                        target = node2, \
                        cutoff= maxl))
        for p in path:
            if len(p) > 0:
                paths.append(p)
    return paths

def sort_paths(graph, paths, sort_by):
    """Takes in a list of paths and sorts them."""

    # Calculate length of path
    lengths = [len(p) for p in paths]
    # Calculate weights of path
    weights = \
    [[graph[p[i]][p[i+1]]["weight"] for i in range(len(p)-1)] for p in paths]
    sum_weights = [np.sum(w) for w in weights]
    avg_weights = [np.mean(w) for w in weights]
    # Sort paths
    paths = zip(paths, lengths, sum_weights, avg_weights)
    if sort_by == "length":
        key = lambda x: x[1]
        reverse = False
    elif sort_by == "cumulative_weight":
        key = lambda x: x[2]
        reverse = True
    elif sort_by == "average_weight":
        key = lambda x: x[3]
        reverse = True
    sorted_paths = sorted(paths, key = key, reverse = reverse)
    return sorted_paths

def get_common_nodes(paths, threshold, maxl):
    """Takes in an list of paths and returns the nodes which are more 
    common than the provided threshold.
    """

    # Convert path list to array where all paths are equal sized
    # Use the maximum path length as an upper bound
    paths_array = np.array([p + [-1]*(maxl-len(p)) for p in paths])
    # Get unique nodes
    unique_nodes = np.unique(paths_array)
    unique_nodes = unique_nodes[unique_nodes > 0]
    # Get node percentage
    node_count = np.array([(node == paths_array).sum() for node in unique_nodes])
    node_perc = node_count/node_count.sum()
    # Select common nodes
    common_nodes = unique_nodes[node_perc > threshold]
    common_nodes = list(common_nodes)
    return common_nodes

def get_common_edges(paths, threshold):
    """Takes in a list of paths and returns a list of edges which are
    more common than the provided threshold.
    """

    # Initialize lists of edges and their count
    edges = []
    counts = []
    for p in paths:
        for i in range(len(p) - 1):
            # For every path find the edges
            edge = (p[i], p[i+1])
            # If edge is already in the list, increase the out
            if edge in edges:
                index = edges.index(edge)
                counts[index] += 1
            # Else add it to the list, and add a count of 1
            else:
                edges.append(edge)
                counts.append(1)
    edges = np.array(edges)
    counts = np.array(counts)
    # Calculate percentage occurance for each path
    perc = counts/counts.sum()
    # Choose paths larger than the threshold
    common_edges = edges[perc > threshold]
    common_edges = list(map(tuple, common_edges))
    return common_edges

def get_metapath(paths, node_threshold, edge_threshold, maxl):
    """Takes in a list of paths, an edge threshoold and a node threshold
    and returns a list of metapaths where each metapath contains all 
    nodes and edges above their respective thresholds.
    """
    
    # Get common nodes
    common_nodes = get_common_nodes(paths, node_threshold, maxl)
    print("common_nodes", common_nodes)
    # Get common edges
    common_edges = get_common_edges(paths, edge_threshold)
    print("common_edges", common_edges)
    common_paths = []
    # Find which paths have the common nodes and edges
    for p in paths:
        # print(p)
        edges = [(p[i], p[i+1]) for i in range(len(p) - 1)]
        # Check if required nodes are in path
        for c_node in common_nodes:
            if c_node in p:
                # Check if required edges are in path
                for c_edge in common_edges:
                    # Add path if not already added
                    if c_edge in edges and p not in common_paths:
                        common_paths.append(p)
    return common_paths


def main():
    ######################### ARGUMENT PARSER #########################
    description = "Path analysis"
    parser = argparse.ArgumentParser(description= description)

    i_helpstr = ".dat file matrix"
    parser.add_argument("-i", "--input-dat",
                        dest = "input_matrix",
                        help= i_helpstr,
                        action= "store",
                        type= str)

    l_default = 10
    l_helpstr = "Maximum path length (see option -p) (default: {:d})"
    parser.add_argument("-l", "--maximum-path-length", \
                        dest = "maxl",
                        default = l_default,
                        type = int,
                        help = l_helpstr.format(l_default))

    r_default  = 1
    r_helpstr = "Residue spacing (see option -p) (default: {:d})"
    parser.add_argument("-r", "--residue-spacing", \
                        dest = "res_space",
                        default = r_default,
                        type = int,
                        help = r_helpstr.format(r_default))

    p_helpstr = "Calculate all simple paths between " \
                "two residues in the graph"
    parser.add_argument("-p", "--all-paths",
                        dest = "do_paths",
                        action = "store_true",
                        default = False,
                        help = p_helpstr)

    s_choices = ["length", "cumulative_weight", "avg_weight"]
    s_default = "length"
    s_helpstr = \
        "How to sort pathways in output. Possible choices are {:s}" \
        " (default: {:s})"
    parser.add_argument("-s", "--sort-paths",
                        dest = "sort_by",
                        choices = s_choices,
                        default = s_default,
                        help = \
                            s_helpstr.format(", ".join(s_choices), s_default))

    a_helpstr = "Source residue for paths calculation (see option -p)"
    parser.add_argument("-a", "--source",
                        dest = "source",
                        default = None,
                        type = int,
                        help = a_helpstr)

    b_helpstr = "Target residue for paths calculation (see option -p)"
    parser.add_argument("-b", "--target",
                        dest = "target",
                        default = None,
                        type = int,
                        help = b_helpstr)

    o_default = "paths.txt"
    o_helpstr = "Output file name"
    parser.add_argument("-o", "--output",
                        dest = "output",
                        default = o_default,
                        help = o_helpstr)

    options = parser.parse_args()

    # Load file
    matrix = np.loadtxt(options.input_matrix)
    graph = nx.Graph(matrix)

    # Choose whether to get shortest paths or all paths
    if options.do_paths:
        paths = get_all_paths(graph = graph, \
                              source = options.source, \
                              target = options.target, \
                              maxl = options.maxl, \
                              res_space = options.res_space)
    else:
        paths = get_shortest_paths(graph = graph, \
                                    source = options.source, \
                                    target = options.target, \
                                    maxl = options.maxl, \
                                    res_space = options.res_space)

    path_table = sort_paths(graph, paths, options.sort_by)

    # Metapath
    metapath = get_metapath(paths, 0.1, 0.1, options.maxl)
    print(metapath)
    #metapath_table = sort_paths()


    # Write file
    with open(options.output, "w") as f:
        for path, length, sum_weight, avg_weight in path_table:
            f.write(f"{path}\t{length}\t{sum_weight}\t{avg_weight}\n")



    #test graph
    # G = nx.Graph()
    # G.add_nodes_from([1,2,3,4,5,6])
    # G.add_edges_from([(2,3), (3,4), (3,6)])
    # P = get_shortest_paths(G, 1, 6, 10, 1)
    # metapath = get_metapath(P, 0.3, 0.3, options.maxl)
    # print(metapath)




if __name__ == "__main__":
    main()