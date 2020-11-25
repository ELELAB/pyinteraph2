import argparse
import logging as log
import numpy as np
import networkx as nx
import itertools

def get_shortest_paths(graph, source, target, maxl, res_space):
    combinations = itertools.combinations(range(source, 
                                                target + 1, 
                                                res_space), 2)
    paths = []
    for node1, node2 in combinations:
        try:
            path = list(nx.algorithms.shortest_paths.generic.all_shortest_paths(graph, node1, node2))
            for p in path:
                if len(p) <= maxl:
                    paths.append(p)
        except nx.NetworkXNoPath:
            log.info(f"No path found between {node1} and {node2}")
    return paths


        
def main():
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


    # add target option

    options = parser.parse_args()

    matrix = np.loadtxt(options.input_matrix)
    graph = nx.Graph(matrix)

    #test graph
    G = nx.Graph()
    G.add_nodes_from([1,2,3,4,5,6])
    G.add_edges_from([(2,3), (3,4), (3,6)])

    print(get_shortest_paths(G, 1, 6, options.maxl, options.res_space))


if __name__ == "__main__":
    main()