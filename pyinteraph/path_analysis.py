import argparse
import logging as log
import numpy as np
import networkx as nx
import itertools

def get_shortest_paths(graph, source, target, maxl, res_space):
    """Find all shortest paths between all combinations of source and
    target.
    """
    
    # Get all combinations
    combinations = itertools.combinations(range(source, 
                                                target + 1, 
                                                res_space), 2)
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


def main():
    ######################### ARGUMENT PARSER #########################
    description = "Find shortest path"
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

    s_choices = ["length", "cumulative_weight", "avg_weight"]
    s_default = "length"
    s_helpstr = \
        "How to sort pathways in output. Possible choices are {:s}" \
        " (default: {:s})"
    parser.add_argument("-s", "--sort-paths",
                        dest = "sort_by",
                        choices = s_choices,
                        default = "length",
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

    options = parser.parse_args()

    # Load file
    matrix = np.loadtxt(options.input_matrix)
    graph = nx.Graph(matrix)
    all_paths = get_shortest_paths(graph = graph, \
                                   source = options.source, \
                                   target = options.target, \
                                   maxl = options.maxl, \
                                   res_space = options.res_space)
    path_table = sort_paths(graph, all_paths, options.sort_by)

    # Write file
    with open("out.txt", "w") as f:
        for path, length, sum_weight, avg_weight in path_table:
            f.write(f"{path}\t{length}\t{sum_weight}\t{avg_weight}\n")



    #test graph
    #G = nx.Graph()
    #G.add_nodes_from([1,2,3,4,5,6])
    #G.add_edges_from([(2,3), (3,4), (3,6)])
    
    #print(graph.nodes)




if __name__ == "__main__":
    main()