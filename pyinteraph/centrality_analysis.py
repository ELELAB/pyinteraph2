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

def get_degree_cent(G):
    degree_dict = nxc.degree_centrality(G)
    return degree_dict

def get_betweeness_cent(G):
    betweeness_dict = nxc.betweenness_centrality(G)
    return betweeness_dict

def get_closeness_cent(G):
    closeness_cent = nxc.closeness_centrality(G)
    return closeness_cent

def get_centrality_dict(cent_list, function_map, graph):
    """
    Returns a dictionary where the key is the name of a centrality 
    measure and the value is a dictionary of centrality values for each
    node. e.g. {degree: {A: 0.1, B:0.7, ...}, betweenness: {...}}
    """

    centrality_dict = {}
    for name in cent_list:
        cent_dict = function_map[name](graph)
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
        cent_array = np.array([val for val in cent_dict.values()])
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

    c_choices = ["all", "degree", "betweenness", "closeness"]
    c_default = None
    c_helpstr = "Select which centrality measures to calculate: " \
                f"{c_choices} (default: {c_default}"
    parser.add_argument("-c", "--centrality",
                        dest = "cent",
                        nargs = "+",
                        choices = c_choices,
                        default = c_default,
                        help =  c_helpstr)

    co_default = "centrality"
    co_helpstr = f"Output file name for centrality measures " \
                 f"(default: {co_default}.txt"
    parser.add_argument("-co", "--centrality-output",
                        dest = "c_out",
                        default = co_default,
                        help = co_helpstr)

    po_helpstr = f"For each centrality measure calculated, create a PDB file " \
                 f"where the bfactor column is replace by the centrality value."
    parser.add_argument("-po", "--pdb_output",
                        dest = "save_pdb",
                        action = "store_true",
                        default = False,
                        help = po_helpstr)

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

    function_map = {'degree': get_degree_cent, 
                    'betweenness': get_betweeness_cent,
                    'closeness': get_closeness_cent}
    
    if args.cent is not None:
        # Get list of all centrality measures
        if "all" in args.cent:
            centrality_names = list(function_map.keys())
        else:
            centrality_names = args.cent
    
        # Get dictionary of centrality values
        centrality_dict = get_centrality_dict(cent_list = centrality_names,
                                              function_map = function_map, 
                                              graph = graph)
        # Save dictionary as table
        write_table(fname = args.c_out,
                    centrality_dict = centrality_dict, 
                    identifiers = identifiers)

        # Write PDB files if request (and if reference provided)
        if args.save_pdb and args.pdb is not None:
            write_pdb_files(centrality_dict = centrality_dict,
                            pdb = args.pdb,
                            fname = args.c_out)
        else:
            # Warn if no PDB provided
            warn_str = "No reference PDB file provided, no PDB files will be "\
                       "created"
            log.warning(warn_str)

if __name__ == "__main__":
    main()
