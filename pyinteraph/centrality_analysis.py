import os
import sys
import argparse
import logging as log
import numpy as np
import networkx as nx
import MDAnalysis as mda
from networkx.algorithms import centrality as nxc
from Bio import PDB
import matplotlib.pyplot as plt
import itertools

def build_graph(fname, pdb = None):
    """Build a graph from the provided matrix"""

    try:
        adj_matrix = np.loadtxt(fname)
    except:
        errstr = f"Could not load file {fname} or wrong file format."
        raise ValueError(errstr)
    # if the user provided a reference structure
    if pdb is not None:
        try:
            # generate a Universe object from the PDB file
            u = mda.Universe(pdb)
        except FileNotFoundError:
            raise FileNotFoundError(f"PDB not found: {pdb}")
        except:
            raise Exception(f"Could not parse pdb file: {pdb}")
        # generate identifiers for the nodes of the graph
        identifiers = [f"{r.segment.segid}{r.resnum}" for r in u.residues]
    # if the user did not provide a reference structure
    else:
        # generate automatic identifiers going from 1 to the
        # total number of residues considered
        identifiers = [f"_{i}" for i in range(1, adj_matrix.shape[0]+1)]

    # generate a graph from the data loaded
    G = nx.Graph(adj_matrix)
    # set the names of the graph nodes (in place)
    node_names = dict(zip(range(adj_matrix.shape[0]), identifiers))
    nx.relabel_nodes(G, mapping = node_names, copy = False)
    # return the identifiers and the graph
    return identifiers, G

def convert_input_to_list(user_input, identifiers):
    """Take in a string (e.g. A12:A22,A13... if a PDB file is supplied)
    and a list of names of all the residues (graph nodes). Replaces the 
    range indicated by the colon with all residues in that range and 
    keeps all residues separated by commas. Removes duplicates.
    """

    res_list = []
    # Split by comma and then colon to make list of list
    split_list = [elem.split(':') for elem in user_input.split(',')]
    for sub_list in split_list:
        # If list size is one, word is separated by comma
        if len(sub_list) == 1:
            try:
                res = sub_list[0]
                # Find index
                identifiers.index(res)
                # Append to list
                res_list.append(res)
            except:
                raise ValueError(f"Residue not in PDB: {res}")
        # If list size is 2, word is separated by a single colon
        elif len(sub_list) == 2:
            try:
                # Get indexes for residue in identifiers
                u_idx, v_idx = [identifiers.index(res) for res in sub_list]
            except:
                raise ValueError(f"Residue range not in PDB: {':'.join(sub_list)}")
            # Check if order of residues is reversed
            if u_idx >= v_idx:
                raise ValueError(f"Range not specified correctly: {':'.join(sub_list)}")
            try:
                # This block should not cause an error
                # Create list with all residues in that range
                res_range = identifiers[u_idx:v_idx + 1]
                # Concatenate with output list
                res_list += res_range
            except Exception as e:
                raise e
        # Other list sizes means multiple colons
        else:
            err_str = f"Incorrect format, only one ':' allowed: {':'.join(sub_list)}"
            raise ValueError(err_str)
    # Remove duplicates
    res_list = list(set(res_list))
    return res_list

def get_hubs(G, **kwargs):
    """Returns a dictionary of degree values for each node"""

    degree_tuple = G.degree()
    hubs = {n : (d if d >= kwargs['hub'] else 0) for n, d in degree_tuple}
    return hubs

def get_degree_cent(G, **kwargs):
    """Returns a dictionary of degree centrality values"""

    centrality_dict = nxc.degree_centrality(G)
    return centrality_dict

def get_betweeness_cent(G, **kwargs):
    """Returns a dictionary of betweeness centrality values"""

    # Need to consider if endpoints should be used or not
    centrality_dict = nxc.betweenness_centrality(G = G,
                                                 normalized = kwargs['norm'],
                                                 weight = kwargs['weight_name'],
                                                 endpoints = kwargs['endpoint'])
    return centrality_dict

def get_closeness_cent(G, **kwargs):
    """Returns a dictionary of closeness centrality values"""

    centrality_dict = nxc.closeness_centrality(G = G)
    return centrality_dict


def get_communicability_betweenness_cent(G, **kwargs):
    """Returns a dictionary of communicability betweenness centrality values"""

    centrality_dict = nxc.communicability_betweenness_centrality(G = G,
                                                                 normalized = kwargs['norm'])
    return centrality_dict

def get_dict_with_group_val(G, node_list, value):
    """Take in a graph, list of nodes and a single value. Returns a dict
    containing each node in the graph. If the node is in the list, its
    value is the given value or else it is 0.
    """

    node_dict = {n : (value if n in node_list else 0) for n in G.nodes()}
    return node_dict


def get_group_betweenness_cent(G, **kwargs):
    """Returns a dictionary of group betweeness centrality values"""

    centrality_val = nxc.group_betweenness_centrality(G = G,
                                                      C = kwargs['node_list'],
                                                      normalized = kwargs['norm'],
                                                      weight = kwargs['weight_name'])
    centrality_dict = get_dict_with_group_val(G, kwargs['node_list'], centrality_val)
    return centrality_dict

def get_group_closeness_cent(G, **kwargs):
    """Returns a dictionary of group closeness centrality values"""

    centrality_val = nxc.group_closeness_centrality(G = G,
                                                    S = kwargs['node_list'],
                                                    weight = kwargs['weight_name'])
    centrality_dict = get_dict_with_group_val(G, kwargs['node_list'], centrality_val)
    return centrality_dict

def get_edge_betweenness_cent(G, **kwargs):
    """Returns a dictionary of edge betweenness centrality values"""
    centrality_dict = nxc.edge_betweenness_centrality(G = G,
                                                      normalized = kwargs['norm'],
                                                      weight = kwargs['weight_name'])
    return centrality_dict

def get_centrality_dict(cent_list, function_map, graph, **kwargs):
    """
    Returns a dictionary where the key is the name of a centrality 
    measure and the value is a dictionary of centrality values for each
    node. e.g. {degree: {A: 0.1, B:0.7, ...}, betweenness: {...}}
    """

    node_dict = {}
    edge_dict = {}
    for name in cent_list:
        cent_dict = function_map[name](G = graph, **kwargs)
        if "edge" in name:
            edge_dict[name] = cent_dict
        else:
            node_dict[name] = cent_dict
    return node_dict, edge_dict

def sort_dictionary(centrality_dict):
    # Only works with one dict, fix for multiple dict
    sorted_dict = {}
    for name, node_dict in centrality_dict.items():
        sorted_node_dict = {n : v for n, v in sorted(node_dict.items(), key = lambda item:item[1], reverse= True)}
        sorted_dict[name] = sorted_node_dict
    return sorted_dict

def write_table(fname, centrality_dict, identifiers):
    """
    Takes in a dictionary of dictionaries and saves a file where each 
    row consists of a node and its corresponding centrality values.
    """

    # Remove any file extensions
    fname = os.path.splitext(fname)[0]

    with open(f"{fname}.txt", "w") as f:
        # Add first line (header)
        line = f"node"
        for key in centrality_dict.keys():
            # Add name of each centrality
            line += f"\t{key}"
        line += "\n"
        f.write(line)
        # Add each row (represents a node)
        first_cent = list(centrality_dict.keys())[0]
        sorted_nodes = [n for n in centrality_dict[first_cent].keys()]
        #for node in identifiers:
        for node in sorted_nodes:
            line = f"{node}"
            for c_dict in centrality_dict.values():
                # Add each centrality vlue
                line += f"\t{c_dict[node]}"
            line += "\n"
            f.write(line)

def replace_bfac_column(pdb, vals, pdb_out):
    """Replace the column containing B-factors in a PDB with
    custom values. Takes in the reference pdb file name, an array of
    values and the output pdb file name
    """

    # create the PDB parser
    parser = PDB.PDBParser()
    # get the protein structure
    structure = parser.get_structure("protein", pdb)
    io = PDB.PDBIO()
    chain_offset = 0
    for model in structure:
        for chain in model:
            for i, residue in enumerate(chain):
                for atom in residue:
                    # set the custom value
                    atom.set_bfactor(float(vals[i+chain_offset]))
            chain_offset += len(chain)
    # set the structure for the output
    io.set_structure(structure)
    # save the structure to a new PDB file
    io.save(pdb_out)

def write_pdb_files(centrality_dict, pdb, fname):
    """Save a pdb file for every centrality measure in the input 
    centrality dictionary
    """

    for cent_name, cent_dict in centrality_dict.items():
        # Create input array
        cent_array = np.array([val for val in cent_dict.values()])
        # Replace column and save PDB file
        replace_bfac_column(pdb, cent_array, f"{cent_name}_{fname}.pdb")

def write_edge_table(fname, centrality_dict, identifiers):
    """
    Takes in a dictionary of dictionaries and saves a file where each 
    row consists of a node and its corresponding centrality values.
    """

    # Remove any file extensions
    fname = os.path.splitext(fname)[0]
    
    # Get sorted list of all edges in the whole dictionary
    all_edges = []
    for inner_dict in centrality_dict.values():
        edges = inner_dict.keys()
        for edge in edges:
            if edge not in all_edges:
                all_edges.append(edge)
    all_edges.sort()

    with open(f"{fname}.txt", "w") as f:
        # Add first line (header)
        line = f"edge"
        for key in centrality_dict.keys():
            # Add name of each centrality
            line += f"\t{key}"
        line += "\n"
        f.write(line)
        #for edge in identifiers:
        for edge in all_edges:
            # Write edge name
            line = f"{edge[0]},{edge[1]}"
            for c_dict in centrality_dict.values():
                # Add entrality value if edge exists in this dict
                if c_dict.get(edge):
                    line += f"\t{c_dict[edge]}"
                # Else add 0
                else:
                    line += "\t0"
            line += "\n"
            f.write(line)

def save_matrix(centrality_dict, identifiers):
    """
    Takes in a dictionary of dictionary and saves a matrix file for
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

    description = "Centrality analysis module for PyInteraph. Allows for " \
                  "calcution of hubs, node centralities, group centralities " \
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

    c_choices = ["node", "degree", "betweenness", "closeness", "communicability",
                 "group", "group_betweenness", "group_closeness",
                 "edge", "edge_betweenness"]
    c_default = None
    c_helpstr = "Select which centrality measures to calculate: " \
                f"{c_choices} (default: {c_default}). Group centralities " \
                "will only be calculated if a list of nodes is provided " \
                "(see option -g)."
    parser.add_argument("-c", "--centrality",
                        dest = "cent",
                        nargs = "+",
                        choices = c_choices,
                        default = c_default,
                        help =  c_helpstr)

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

    g_helpstr = "List of residues used for group centrality calculations. " \
                "e.g. A32,A35,A37:A40. Replace chain name with '_' if no " \
                "reference PDB file provided. e.g. _42,_57."
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
                f"where the bfactor column is replace by the centrality value." \
                f" (default: {p_default})."
    parser.add_argument("-p", "--pdb_output",
                        dest = "save_pdb",
                        action = "store_true",
                        default = False,
                        help = p_helpstr)

    m_default = False
    m_helpstr = f"For each edge centrality measure calculated, create a .dat " \
                f"file (matrix) containing the centrality values. " \
                f" (default: {m_default})."
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
    identifiers, graph = build_graph(fname = args.input_matrix,
                                     pdb = args.pdb)

    # get graph nodes and edges
    nodes = graph.nodes()
    edges = graph.edges()
    # print nodes
    info = f"Graph loaded! {len(nodes)} nodes, {len(edges)} edges\n" \
           f"Node list:\n{np.array(identifiers)}\n"
    sys.stdout.write(info)

    ############################ CENTRALITY ############################

    # Centrality types
    node = ["hubs", "degree", "betweenness", "closeness"] # add communicability
    group = ["group_betweenness", "group_closeness"]
    edge = ["edge_betweenness"]
    all_cent = node + edge

    # Function map of all implemented measures
    function_map = {'hubs' : get_hubs,
                    'degree': get_degree_cent, 
                    'betweenness': get_betweeness_cent,
                    'closeness': get_closeness_cent,
                    #'communicability' : get_communicability_betweenness_cent,
                    'group_betweenness' : get_group_betweenness_cent,
                    'group_closeness' : get_group_closeness_cent,
                    'edge_betweenness' : get_edge_betweenness_cent}
    
    # Get list of all centrality measures
    if args.cent is not None:
        # Get node list if present
        if args.group is not None:
            node_list = convert_input_to_list(args.group, identifiers)
        else:
            node_list = args.group

        # Find group names based on user request
        # Find all non group centralities
        if "all" in args.cent:
            centrality_names = all_cent
        # Find all node centralities
        if "node" in args.cent:
            centrality_names = node
        # Find all group centralities if node list specified
        elif "group" in args.cent and args.group is not None:
            centrality_names = group
        # Throw error if no group is requested but node list not specified
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

        # Change weight boolean to weight name or None
        if args.weight is False:
            args.weight = None
        else:
            args.weight = "weight"

        # Print message
        sys.stdout.write("Calculating:\n")
        for name in centrality_names:
            sys.stdout.write(f"{name} centrality\n")

        # Create dictionary of optional arguments
        kwargs = {'node_list' : node_list,
                  'weight_name' : args.weight,
                  'norm' : args.norm,
                  'endpoint' : args.endpoint,
                  'hub': args.hub}

        # node = any([True for n in centrality_names if n in node or n in group])
        # edge = any([True for n in centrality_names if n in edge])
        # print(node)
        
        # Get dictionary of node/group centrality values
        node_dict, edge_dict = get_centrality_dict(cent_list = centrality_names,
                                                   function_map = function_map, 
                                                   graph = graph,
                                                   **kwargs)

        # Dictionary is not empty so node centralities have been requested
        if node_dict:
            # Convert dictionary to sorted list ??? this is still a dict
            node_dict = sort_dictionary(node_dict)

            # Save dictionary as table
            write_table(fname = args.c_out,
                        centrality_dict = node_dict, 
                        identifiers = identifiers)

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
            write_edge_table(fname = f"{args.c_out}_edge",
                             centrality_dict = edge_dict, 
                             identifiers =  identifiers)
            if args.save_mat:
                save_matrix(centrality_dict = edge_dict, 
                            identifiers = identifiers)


    # Delete later
    # print(node_dict['betweenness']['_99'])
    # x = nx.Graph()
    # a = [('A', 'B'), ('B', 'C'), ('C','D')]
    # b = [('A', 'B'), ('B', 'C'), ('C','D'), ('B', 'E')]
    # c = [('A', 'B'), ('B', 'C'), ('C','D'), ('B', 'E'), ('C', 'E')]
    # d = [('A', 'B'), ('B', 'C'), ('C','A')]
    # x.add_edges_from(b)
    # print(x.degree())
    # print(x.edges())
    # print(nxc.betweenness_centrality(x))
    # print(nxc.betweenness_centrality(x, endpoints = True))
    #print(nxc.betweenness_centrality(x))
    #print(nx.algorithms.centrality.group_betweenness_centrality(x, C=[0,3]))

    # #plot graph
    # weights = [d["weight"] for u, v, d in graph.edges(data=True)]
    # pos = nx.spring_layout(graph, k = 0.5)
    # nx.draw_networkx_nodes(graph, pos)
    # nx.draw_networkx_edges(graph, pos, edge_color = weights)
    # nx.draw_networkx_labels(graph, pos)
    # plt.savefig("graph.png")

if __name__ == "__main__":
    main()
