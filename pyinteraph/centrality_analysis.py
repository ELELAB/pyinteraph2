import os
import sys
import argparse
import logging as log
import numpy as np
import networkx as nx
import MDAnalysis as mda
from networkx.algorithms import centrality as nxc
from Bio import PDB

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
        identifiers = [str(i) for i in range(1, adj_matrix.shape[0]+1)]

    # generate a graph from the data loaded
    G = nx.Graph(adj_matrix)
    # set the names of the graph nodes (in place)
    node_names = dict(zip(range(adj_matrix.shape[0]), identifiers))
    nx.relabel_nodes(G, mapping = node_names, copy = False)
    # return the idenfiers and the graph
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

def get_degree_cent(G, node_list, weight_name, norm):
    """Returns a dictionary of degree centrality values"""

    centrality_dict = nxc.degree_centrality(G)
    return centrality_dict

def get_betweeness_cent(G, node_list, weight_name, norm):
    """Returns a dictionary of betweeness centrality values"""

    # Need to consider if endpoints should be used or not
    centrality_dict = nxc.betweenness_centrality(G = G,
                                                 normalized = norm,
                                                 weight = weight_name)
    return centrality_dict

def get_closeness_cent(G, node_list, weight_name, norm):
    """Returns a dictionary of closeness centrality values"""

    centrality_dict = nxc.closeness_centrality(G = G)
    return centrality_dict


def get_communicability_betweenness_centrality(G, node_list, weight_name, norm):
    """Returns a dictionary of communicability betweenness centrality values"""

    centrality_dict = nxc.communicability_betweenness_centrality(G = G,
                                                                 normalized = norm)
    return centrality_dict

def get_group_betweenness_cent(G, node_list, weight_name, norm):
    """Returns a dictionary of group betweeness centrality values"""

    centrality_val = nxc.group_betweenness_centrality(G = G,
                                                      C = node_list,
                                                      normalized = norm,
                                                      weight = weight_name)
    centrality_dict = {}
    for node in G.nodes():
        if node in node_list:
            centrality_dict[node] = centrality_val
        else:
            centrality_dict[node] = 0
    return centrality_dict

def get_group_closeness_cent(G, node_list, weight_name, norm):
    """Returns a dictionary of group closeness centrality values"""

    centrality_val = nxc.group_closeness_centrality(G = G,
                                                    S = node_list,
                                                    weight = weight_name)
    centrality_dict = {}
    for node in G.nodes():
        if node in node_list:
            centrality_dict[node] = centrality_val
        else:
            centrality_dict[node] = 0
    return centrality_dict

def get_centrality_dict(cent_list, function_map, graph, node_list, weight_name, norm):
    """
    Returns a dictionary where the key is the name of a centrality 
    measure and the value is a dictionary of centrality values for each
    node. e.g. {degree: {A: 0.1, B:0.7, ...}, betweenness: {...}}
    """

    centrality_dict = {}
    for name in cent_list:
        cent_dict = function_map[name](G = graph, 
                                       node_list = node_list,
                                       weight_name = weight_name, 
                                       norm = norm)
        centrality_dict[name] = cent_dict
    return centrality_dict


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
        for node in identifiers:
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


def main():

    ######################### ARGUMENT PARSER #########################

    description = "Path analysis"
    parser = argparse.ArgumentParser(description= description)

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
                 "edge"]
    c_default = None
    c_helpstr = "Select which centrality measures to calculate: " \
                f"{c_choices} (default: {c_default} Group centralities " \
                "will only be calcu"
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

    g_helpstr = "List of residues used for group centrality calculations. " \
                "e.g. A32,A35,A37:A40. Replace chain name with '_' if no " \
                "reference PDB file provided. e.g. _42,_57"
    parser.add_argument("-g", "--group",
                        dest = "group",
                        default = None,
                        type = str,
                        help = g_helpstr)

    o_default = "centrality"
    o_helpstr = f"Output file name for centrality measures " \
                f"(default: {o_default}.txt"
    parser.add_argument("-o", "--centrality-output",
                        dest = "c_out",
                        default = o_default,
                        help = o_helpstr)

    p_default = False
    p_helpstr = f"For each centrality measure calculated, create a PDB file " \
                f"where the bfactor column is replace by the centrality value." \
                f" (default: {p_default}"
    parser.add_argument("-p", "--pdb_output",
                        dest = "save_pdb",
                        action = "store_true",
                        default = False,
                        help = p_helpstr)

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

    # Groups
    node = ["degree", "betweenness", "closeness"]
    group = ["group_betweenness", "group_closeness"]
    edge = []

    # Function map of all implemented measures
    function_map = {'degree': get_degree_cent, 
                    'betweenness': get_betweeness_cent,
                    'closeness': get_closeness_cent,
                    'communicability' : get_communicability_betweenness_centrality,
                    'group_betweenness' : get_group_betweenness_cent,
                    'group_closeness' : get_group_closeness_cent}
    
    # Get list of all centrality measures
    if args.cent is not None:
        # Get node list if present
        if args.group is not None:
            node_list = convert_input_to_list(args.group, identifiers)
        else:
            node_list = args.group

        # Find all node centralities
        if "node" in args.cent:
            centrality_names = node
        # Find all group centralities if group specified
        elif "group" in args.cent and args.group is not None:
            centrality_names = group
        # Throw error if no group is requested but not specified
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
            log.error("Not implemented")
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
        
        # Get dictionary of centrality values
        centrality_dict = get_centrality_dict(cent_list = centrality_names,
                                              function_map = function_map, 
                                              graph = graph,
                                              node_list = node_list,
                                              weight_name = args.weight,
                                              norm = args.norm)
        # Save dictionary as table
        write_table(fname = args.c_out,
                    centrality_dict = centrality_dict, 
                    identifiers = identifiers)

        # Write PDB files if request (and if reference provided)
        if args.save_pdb and args.pdb is not None:
            write_pdb_files(centrality_dict = centrality_dict,
                            pdb = args.pdb,
                            fname = args.c_out)
        elif args.pdb is None:
            # Warn if no PDB provided
            warn_str = "No reference PDB file provided, no PDB files will be "\
                       "created."
            log.warning(warn_str)


        # Delete later
        x = nx.Graph()
        y = [(0, 1), (1, 2), (2,3)]
        x.add_edges_from(y)
        print(x.degree())
        #print(nxc.betweenness_centrality(x))
        #print(nx.algorithms.centrality.group_betweenness_centrality(x, C=[0,3]))

if __name__ == "__main__":
    main()
