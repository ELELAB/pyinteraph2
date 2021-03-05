import os
import sys
import argparse
import logging as log
import numpy as np
import networkx as nx
from networkx.algorithms import centrality as nxc
import pandas as pd
import graph_analysis as ga
import path_analysis as pa

def get_hubs(G, **kwargs):
    """Returns a dictionary of degree values for each node."""

    degree_tuple = G.degree()
    hubs = {n : (d if d >= kwargs["hub"] else 0) for n, d in degree_tuple}
    return hubs

def get_degree_cent(G, **kwargs):
    """Returns a dictionary of degree centrality values for each node."""

    centrality_dict = nxc.degree_centrality(G)
    return centrality_dict

def get_betweeness_cent(G, **kwargs):
    """Returns a dictionary of betweeness centrality values for each node."""

    # Need to consider if endpoints should be used or not
    centrality_dict = nxc.betweenness_centrality(G = G,
                                                 normalized = kwargs["normalized"],
                                                 weight = kwargs["weight"],
                                                 endpoints = kwargs["endpoints"])
    return centrality_dict

def get_closeness_cent(G, **kwargs):
    """Returns a dictionary of closeness centrality values for each node."""

    centrality_dict = nxc.closeness_centrality(G = G)
    return centrality_dict

def remove_isolates(G):
    """Takes in a graph and returns a new graph where all the nodes with 
    zero edges have been removed.
    """

    # create duplicate graph so the original is unaffected
    H = G.copy()
    # isolates are nodes with zero edges
    isolates = nx.isolates(G)
    # remove from duplicate graph
    H.remove_nodes_from(list(isolates))
    return H

def fill_dict_with_isolates(G, cent_dict):
    """Takes in a graph and a dictionary of centrality values. Returns a
    new dictionary of centrality values containing each node in the 
    graph. If the node is not in the given dictionary, it has a value of
    0 or else if has the value in the given dictionary.
    """

    return {n : (cent_dict[n] if n in cent_dict else 0) for n in G.nodes()}

def get_communicability_betweenness_cent(G, **kwargs):
    """Returns a dictionary of communicability betweenness centrality values"""

    G_no_isolates = remove_isolates(G)
    centrality_dict = nxc.communicability_betweenness_centrality(\
                                G = G_no_isolates,
                                normalized = kwargs["normalized"])
    filled_dict = fill_dict_with_isolates(G, centrality_dict)
    return filled_dict

def get_current_flow_betweenness_cent(G, **kwargs):
    """Returns a dictionary of current flow betweenness centrality
    values. This is the same as calculating betweenness centrality using
    random walks instead of shortest paths.
    """

    G_no_isolates = remove_isolates(G)
    centrality_dict = nxc.current_flow_betweenness_centrality(\
                                G = G_no_isolates, 
                                normalized = kwargs["normalized"],
                                weight = kwargs["weight"])
    filled_dict = fill_dict_with_isolates(G, centrality_dict)
    return filled_dict

def get_dict_with_group_val(G, node_list, value):
    """Takes in a graph, list of nodes and a group value. Returns a dict
    containing each node in the graph. If the node is in the list, its
    value is the group value or else it is 0.
    """

    return {n : (value if n in node_list else 0) for n in G.nodes()}

def get_group_betweenness_cent(G, **kwargs):
    """Returns a dictionary of group betweeness centrality values."""

    centrality_val = nxc.group_betweenness_centrality(G = G,
                                                      C = kwargs["node_list"],
                                                      normalized = kwargs["normalized"],
                                                      weight = kwargs["weight"])
    centrality_dict = get_dict_with_group_val(G, kwargs["node_list"], centrality_val)
    return centrality_dict

def get_group_closeness_cent(G, **kwargs):
    """Returns a dictionary of group closeness centrality values."""

    centrality_val = nxc.group_closeness_centrality(G = G,
                                                    S = kwargs["node_list"],
                                                    weight = kwargs["weight"])
    centrality_dict = get_dict_with_group_val(G, kwargs["node_list"], centrality_val)
    return centrality_dict

def reorder_edge_names(edge_dict):
    """Takes in a dictionary where the keys are edge names and returns 
    a dictionary with where the keys are sorted egde names.
    """

    # lambda function takes in an edge e.g. ("A99", "A102"), removes all
    # non numeric characters and converts to an integer
    node_to_int = lambda node: int(''.join([n for n in node if n.isdigit()]))
    # sort the edge names by their corresponding node number (99 and 102) 
    # and convert the sorted edge name to a string
    reordered_dict = {",".join(sorted(edge, key = node_to_int)): value \
                        for edge, value in edge_dict.items()}
    return reordered_dict

def get_edge_betweenness_cent(G, **kwargs):
    """Returns a dictionary of edge betweenness centrality values."""
    
    centrality_dict = nxc.edge_betweenness_centrality(G = G,
                                                      normalized = kwargs["normalized"],
                                                      weight = kwargs["weight"])
    reordered_dict = reorder_edge_names(centrality_dict)
    return reordered_dict

def get_edge_current_flow_betweenness_cent(G, **kwargs):
    """Returns a dictionary of edge current flowbetweenness centrality 
    values. This is the same as calculating edge betweenness centrality 
    using random walks instead of shortest paths.
    """
    
    G_no_isolates = remove_isolates(G)
    centrality_dict = nxc.edge_current_flow_betweenness_centrality(\
                                    G = G_no_isolates,
                                    normalized = kwargs["normalized"],
                                    weight = kwargs["weight"])
    reordered_dict = reorder_edge_names(centrality_dict)
    return reordered_dict

def get_centrality_dict(cent_list, function_map, graph, **kwargs):
    """Returns two dictionaries. For the first dictionary, the key is the 
    name of a node centrality measure and the value is a dictionary of 
    centrality values for each node. (Also includes group centralities)
    e.g. {degree: {A: 0.1, B:0.7, ...}, betweenness: {...}, ...}
    For the second dictionary, the key is the name of an edge centrality
    measure and the value is a dictionary of centrality values for each
    edge, similar to the first dictionary.
    """

    node_dict = {}
    edge_dict = {}
    sys.stdout.write("Calculating:\n")
    for name in cent_list:
        # Print which measure is being calculated
        if name == "hubs":
            sys.stdout.write(f"{name}\n")
        else:
            p_name = name.replace("_", " ")
            sys.stdout.write(f"{p_name} centrality\n")
        # Get dictionary using the function map
        cent_dict = function_map[name](G = graph, **kwargs)
        # Add edge centralities to edge dict
        if "edge" in name:
            edge_dict[name] = cent_dict
        # Add node centralities to node dict
        else:
            node_dict[name] = cent_dict
    return node_dict, edge_dict

def write_table(fname, centrality_dict, identifiers, sort_by):
    """Takes in a dictionary of dictionaries and saves a file where each 
    row consists of a node and its corresponding centrality values.
    """

    # Remove any file extensions
    fname = os.path.splitext(fname)[0]
    # Transform dict to df
    table = pd.DataFrame(centrality_dict)
    # Check row type and change row names
    row_name = None
    if "," in table.index[0]:
        row_name = "edge"
    else:
        row_name = "node"
    table = table.rename_axis(row_name)
    # Choose whether to sort ascending or descending
    ascending_bool = None
    if sort_by == "node" or sort_by == "edge":
        ascending_bool = True
    else:
        ascending_bool = False
    table = table.sort_values(by=[sort_by], ascending = ascending_bool)
    # Save file
    table.to_csv(f"{fname}.txt", sep = "\t")

def write_pdb_files(centrality_dict, pdb, fname):
    """Save a pdb file for every centrality measure in the input 
    centrality dictionary.
    """

    for cent_name, cent_dict in centrality_dict.items():
        # Create input array
        cent_array = np.array([val for val in cent_dict.values()])
        # Replace column and save PDB file
        ga.replace_bfac_column(pdb, cent_array, f"{cent_name}_{fname}.pdb")

def save_matrix(centrality_dict, identifiers):
    """Takes in a dictionary of dictionary and saves a matrix file for
    each inner dictionary.
    """

    for name, edge_dict in centrality_dict.items():
        # Create a graph for each inner dictionary
        G = nx.Graph()
        G.add_nodes_from(identifiers)
        # create list of (node1, node2, edge weight)
        edges = [(edge[0], edge[1], cent) for edge, cent in edge_dict.items()]
        G.add_weighted_edges_from(edges)
        # Convert graph to matrix
        matrix = nx.to_numpy_matrix(G)
        np.savetxt(f"{name}.dat", matrix)

def main():

    ######################### ARGUMENT PARSER #########################

    description = "Centrality analysis module for PyInteraph. It can be used " \
                  "to calculate hubs, node centralities, group centralities " \
                  "and edge centralities."
    parser = argparse.ArgumentParser(description = description)

    i_helpstr = ".dat file matrix"
    parser.add_argument("-i", "--input-dat",
                        dest = "input_matrix",
                        help = i_helpstr,
                        type = str)

    r_helpstr = "Reference PDB file"
    parser.add_argument("-r", "--pdb",
                        dest = "pdb",
                        help = r_helpstr,
                        default = None,
                        type = str)

    # Centrality types
    node = ["hubs", "degree", "betweenness", "closeness", 
            "communicability_betweenness", "current_flow_betweenness"]
    group = ["group_betweenness", "group_closeness"]
    edge = ["edge_betweenness", "edge_current_flow_betweenness"]
    all_cent = node + edge

    c_choices = all_cent + group + ["all", "node", "edge", "group"]
    c_default = None
    c_helpstr = f"Select which centrality measures to calculate: {c_choices} " \
                f"(default: {c_default}). Selecting 'node' will calculate the "\
                f"following centralities: {node}. Selecting 'edge' will " \
                f"calculate the following centralities: {edge}. Selecting " \
                f"'group' will calculate the following centralities: {group}. " \
                f"Note: Group centralities will only be calculated if a list of " \
                f"nodes is provided (see option -g). Selecting 'all' will " \
                f"calculate all non-group centralities."
    parser.add_argument("-c", "--centrality",
                        dest = "cent",
                        nargs = "+",
                        choices = c_choices,
                        default = c_default,
                        help =  c_helpstr)

    b_choices = node + group
    b_default = "node"
    b_helpstr = f"Sort node centralities. Use the name of the " \
                f"desired measure. The name must match one of the names " \
                f"used in option -c. (default: {b_default})."
    parser.add_argument("-b", "--sort-node",
                        dest = "sort_node",
                        choices = b_choices,
                        default = b_default,
                        help = b_helpstr)

    d_choices = edge
    d_default = "edge"
    d_helpstr = f"Sort edge centralities. Use the name of the " \
                f"desired measure. The name must match one of the names " \
                f"used in option -c. (default: {d_default})."
    parser.add_argument("-d", "--sort-edge",
                        dest = "sort_edge",
                        choices = d_choices,
                        default = d_default,
                        help = d_helpstr)

    w_default = False
    w_helpstr = f"Use edge weights to calculate centrality measures. " \
                f"(default: {w_default})."
    parser.add_argument("-w", "--use_weights",
                        dest = "weight",
                        action = "store_true",
                        default = w_default,
                        help = w_helpstr)

    n_default = True
    n_helpstr = f"Normalize centrality measures. " \
                f"(default: {n_default})."
    parser.add_argument("-n", "--normalize",
                        dest = "norm",
                        action = "store_true",
                        default = n_default,
                        help = n_helpstr)

    e_default = False
    e_helpstr = f"Use endpoints when calculating centrality measures. " \
                f"(default: {e_default})."
    parser.add_argument("-e", "--use_endpoints",
                        dest = "endpoint",
                        action = "store_true",
                        default = e_default,
                        help = e_helpstr)

    k_default = 3
    k_helpstr = f"The minimum cutoff for a node to be considered a hub. " \
                f"(default: {c_default})."
    parser.add_argument("-k", "--hub-cutoff",
                        dest = "hub",
                        default = k_default,
                        type = int,
                        help = k_helpstr)

    g_helpstr = f"List of residues used for group centrality calculations. " \
                f"e.g. A32,A35,A37:A40. Replace chain name with '_' if no " \
                f"reference PDB file provided. e.g. _42,_57."
    parser.add_argument("-g", "--group",
                        dest = "group",
                        default = None,
                        type = str,
                        help = g_helpstr)

    o_default = "centrality"
    o_helpstr = f"Output file name for centrality measures " \
                f"(default: {o_default}.txt)."
    parser.add_argument("-o", "--centrality-output",
                        dest = "c_out",
                        default = o_default,
                        help = o_helpstr)

    p_default = False
    p_helpstr = f"For each centrality measure calculated, create a PDB file " \
                f"where the bfactor column is replace by the centrality value. " \
                f"(default: {p_default})."
    parser.add_argument("-p", "--pdb_output",
                        dest = "save_pdb",
                        action = "store_true",
                        default = False,
                        help = p_helpstr)

    m_default = False
    m_helpstr = f"For each edge centrality measure calculated, create a .dat " \
                f"file (matrix) containing the centrality values. " \
                f"(default: {m_default})."
    parser.add_argument("-m", "--matrix_output",
                        dest = "save_mat",
                        action = "store_true",
                        default = False,
                        help = m_helpstr)

    args = parser.parse_args()


    # Check user input
    if not args.input_matrix:
        # exit if the adjacency matrix was not speficied
        log.error("Graph adjacency matrix must be specified. Exiting ...")
        exit(1)

    # Load file, build graphs and get identifiers for graph nodes
    identifiers, graph = pa.build_graph(fname = args.input_matrix,
                                     pdb = args.pdb)

    # get graph nodes and edges
    nodes = graph.nodes()
    edges = graph.edges()
    # print nodes
    info = f"Graph loaded! {len(nodes)} nodes, {len(edges)} edges\n" \
           f"Node list:\n{np.array(identifiers)}\n"
    sys.stdout.write(info)

    ############################ CENTRALITY ############################

    # Function map of all implemented measures
    function_map = {
        "hubs" : get_hubs,
        "degree" : get_degree_cent, 
        "betweenness" : get_betweeness_cent,
        "closeness" : get_closeness_cent,
        "communicability_betweenness" : get_communicability_betweenness_cent,
        "current_flow_betweenness" : get_current_flow_betweenness_cent,
        "group_betweenness" : get_group_betweenness_cent,
        "group_closeness" : get_group_closeness_cent,
        "edge_betweenness" : get_edge_betweenness_cent,
        "edge_current_flow_betweenness" : get_edge_current_flow_betweenness_cent
        }
    
    # Get list of all centrality measures
    if args.cent is not None:
        # Get node list if present
        if args.group is not None:
            node_list = pa.convert_input_to_list(args.group, identifiers)
        else:
            node_list = args.group

        # Find group names based on user request
        # Find all non group centralities
        if "all" in args.cent:
            centrality_names = all_cent
        # Find all node centralities
        elif "node" in args.cent:
            centrality_names = node
        # Find all group centralities if node list specified
        elif "group" in args.cent and args.group is not None:
            centrality_names = group
        # Throw error if group is requested but node list is not specified
        elif ("group" in args.cent or \
            # One of the group centralities is requested
            len([cent for cent in args.cent if cent in group]) > 0) \
            and args.group is None:
            error_str = "A group of residues must be specified to calculate " \
                        "group centrality (see option -g). Exiting..."
            log.error(error_str)
            exit(1)
        # Find all edge centralities
        elif "edge" in args.cent:
            centrality_names = edge
        else:
            # Get list of specified centrality names
            centrality_names = args.cent

        # Check sorting options
        # sorting arg is either node or in centrality names
        if not (args.sort_node == "node" or args.sort_node in centrality_names):
            # expected node names
            expected_names = [name for name in centrality_names if name not in edge]
            err_str = f"The node sorting centrality argument (option -b) must " \
                      f"be one of the following: {', '.join(expected_names)}. Exiting..."
            log.error(err_str)
            exit(1)

        # sorting arg is either edge or in centrality names
        if not (args.sort_edge == "edge" or args.sort_edge in centrality_names):
            # expected edge names
            expected_names = [name for name in centrality_names if name not in node]
            err_str = f"The edge sorting centrality argument (option -d) must " \
                      f"be one of the following: {', '.join(expected_names)}. Exiting..."
            log.error(err_str)
            exit(1)

        # Change weight boolean to weight name or None
        if args.weight is False:
            args.weight = None
        else:
            args.weight = "weight"

        # Create dictionary of optional arguments
        kwargs = {"node_list" : node_list,
                  "weight" : args.weight,
                  "normalized" : args.norm,
                  "endpoints" : args.endpoint,
                  "hub": args.hub}
        
        # Get dictionary of node+group and edge centrality values
        node_dict, edge_dict = get_centrality_dict(cent_list = centrality_names,
                                                   function_map = function_map, 
                                                   graph = graph,
                                                   **kwargs)

        # Dictionary is not empty so node centralities have been requested
        if node_dict:
            # Save dictionary as table
            write_table(fname = args.c_out,
                        centrality_dict = node_dict, 
                        identifiers = identifiers,
                        sort_by = args.sort_node)

            # Write PDB files if request (and if reference provided)
            if args.save_pdb and args.pdb is not None:
                write_pdb_files(centrality_dict = node_dict,
                                pdb = args.pdb,
                                fname = args.c_out)
            elif args.pdb is None:
                # Warn if no PDB provided
                warn_str = "No reference PDB file provided, no PDB files will "\
                           "be created."
                log.warning(warn_str)

        # Dictionary is not empty so edge centralities have been requested
        if edge_dict:
            write_table(fname = f"{args.c_out}_edge",
                             centrality_dict = edge_dict, 
                             identifiers =  identifiers,
                             sort_by = args.sort_edge)
            if args.save_mat:
                save_matrix(centrality_dict = edge_dict, 
                            identifiers = identifiers)

        # For testing, will delete after all measures implemented
        # x = nx.Graph()
        # a = [('A', 'B'), ('B', 'C'), ('C','D')]
        # b = [('A', 'B'), ('B', 'C'), ('C','D'), ('B', 'E')]
        # c = [('A', 'B'), ('B', 'C'), ('C','D'), ('B', 'E'), ('C', 'E')]
        # d = [('A', 'B'), ('B', 'C'), ('C','A')]
        # e = [('A', 'B'), ('B', 'C'), ('C','D'), ('D', 'A')]
        # x.add_edges_from(e)
        # x.add_node('Z')
        # print(x.degree())
        # print(x.edges())
        # print("BC:", nxc.betweenness_centrality(x))
        # print("BC_e:", nxc.betweenness_centrality(x, endpoints = True))
        # print("CBC:", nxc.communicability_betweenness_centrality(x))
        # #print("CFB:", nxc.current_flow_betweenness(x))
        # print("CFE:", nxc.edge_current_flow_betweenness_centrality(x))
if __name__ == "__main__":
    main()
