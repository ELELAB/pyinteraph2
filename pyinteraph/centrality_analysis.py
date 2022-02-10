#!/usr/bin/env python

#   centrality_analysis.py: script to calculate centrality properties
#                           from network
#   Copyright (C) 2021 Mahdi Robbani, Valentina Sora, Matteo Tiberti
#   Elena Papaleo
#
#    This program is free software: you can redistribute it
#    and/or modify it under the terms of the GNU General Public
#    License as published by the Free Software Foundation, either
#    version 3 of the License, or (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.
#    If not, see <http://www.gnu.org/licenses/>.

# Standard library
import argparse
import logging as log
import os
import sys
# Third-party packages
import MDAnalysis as mda
import networkx as nx
from networkx.algorithms import centrality as nxc
import numpy as np
import pandas as pd
# pyinteraph
from pyinteraph import graph_analysis as ga
from pyinteraph import path_analysis as pa



########################### HELPER FUNCTIONS ##########################



def get_graph_without_glycine(G, identifiers, residue_names):
    """Takes in a graph, a list of nodes for the graph and a list of
    residues corresponding to each node. Returns a graph without any
    nodes corresponding to a glycine.
    """
    
    # Get a dictionary mapping each node to its corresponding residue
    node_dict = dict(zip(identifiers, residue_names))
    
    # Get all nodes which correspond to a glycine
    glycine_nodes = \
        [node for node, res in node_dict.items() if res == "GLY"]
    
    # Create a new graph without glycines
    H = G.copy()
    H.remove_nodes_from(glycine_nodes)

    # Return the new graph
    return H


def finalize_dict(G, cent_dict):
    """Takes in a graph and a dictionary of centrality values.
    Returns a new dictionary mapping each node in the graph with
    its centrality values. If the node is not in the given 
    dictionary, it will be assigned a value of 0.
    """

    return {n : (cent_dict[n] if n in cent_dict else 0) \
            for n in G.nodes()}


def reorder_edge_names(edge_dict):
    """Takes in a dictionary where the keys are edge names and returns
    a dictionary with where the keys are sorted edge names.
    """

    # Function that takes in an edge (e.g. "A99", "A102"), removes
    # non numeric characters and converts the numeric ones
    # to integers
    convert_node_to_int = \
        lambda node: int("".join([n for n in node if n.isdigit()]))
    
    # Sort the edge names by their corresponding node number
    # (e.g. 99 and 102) and convert the sorted edge name
    # to a string
    return \
        {",".join(sorted(edge, key = convert_node_to_int)) : value \
                         for edge, value in edge_dict.items()}


def get_centrality_dict(centrality_list, name2function, graph, **kwargs):
    """Returns two dictionaries. In the first dictionary, keys are
    the names of centrality measures and values are dictionaries of 
    centrality values for each node e.g. {degree: {A: 0.1, B:0.7, ...},
    betweenness: {...}, ...}. In the second dictionary, keys are the 
    names of edge centrality measures and values are dictionaries of
    centrality values for each edge, similar to the first dictionary.
    """

    # List of measures that require a connected graph
    connected_measures = \
        ["current_flow_betweenness", "current_flow_closeness", 
         "edge_current_flow_betweenness"]
    
    # Get all components
    components = nx.algorithms.components.connected_components(graph)
    
    # Sort components by size (biggest first)
    components = sorted(components, 
                        key = lambda x: len(x),
                        reverse = True)
    
    # List of subgraphs for each connected component with at least
    # 3 nodes
    subgraphs = \
        [graph.subgraph(component).copy() for component in components
         if len(component) > 2]
    
    # Intialize output dictionaries
    node_dict = {}
    edge_dict = {}
    identifiers = kwargs["identifiers"]
    res_names = kwargs["residue_names"]
    
    # Add residue names to node_dict
    node_dict["name"] = \
        {identifiers[i]: res_names[i] for i in range(len(identifiers))}
    
    # Write out that the calculation has started
    sys.stdout.write("Calculating:\n")
    
    # For each measure in the list of measures that need to be
    # calculated
    for name in centrality_list:
        
        # Write out which measure is being calculated
        sys.stdout.write(f"{name.replace('_', ' ').capitalize()}\n")
        
        # Choose whether to insert to node_dict or edge_dict
        insert_dict = edge_dict if "edge" in name else node_dict
        
        #------------ Measures requiring a connected graph -----------#
        
        if name in connected_measures:
            
            for n, subgraph in enumerate(subgraphs):
                
                # Calculate the centrality values
                cent_dict = name2function[name](G = subgraph, **kwargs)
                
                # Finalize the dictionary
                if "edge" not in name:
                    cent_dict = finalize_dict(graph, cent_dict)
                
                # Add to dictionary with a name corresponding
                # to the subgraph
                insert_dict[f"{name}_{n+1}"] = cent_dict

        #---------- Measures not requiring a connected graph ---------#

        else:
            # Calculate the centrality values
            cent_dict = name2function[name](G = graph, **kwargs)

            # Add to dictionary with a name corresponding
            # to the subgraph
            insert_dict[name] = cent_dict
    
    # Return the dictionaries with node centralities' values and
    # edge centralities' values
    return node_dict, edge_dict


def write_table(fname, centrality_dict, sort_by):
    """Takes in a dictionary of dictionaries and saves a CSV file
    containing a dataframe where each row consists of a node/edge
    and each column correspond to a centrality value for that
    node/edge.
    """
    
    # Transform the dictionary to a dataframe
    table = pd.DataFrame(centrality_dict)
    
    # Find if the row contains nodes or edges
    row_name = "edge" if "," in table.index[0] else "node"
    table = table.rename_axis(row_name)
    
    # Only sort if node/edge is not specified
    if not(sort_by == "node" or sort_by == "edge"):
        table = table.sort_values(by = [sort_by],
                                  ascending = False)
    
    # Save file
    table.to_csv(fname, sep = ",", na_rep = "NA") 


def write_pdb_files(centrality_dict_node, pdb, fname):
    """Save a pdb file for every centrality measure in the input 
    centrality dictionary.
    """

    for cent_name, cent_dict in centrality_dict_node.items():
        
        # Ignore residue name column
        if cent_name != "name":
            
            # Create input array
            cent_array = np.array([val for val in cent_dict.values()])
            
            # Replace the B-factor column and save the PDB file
            ga.replace_bfac_column(pdb,
                                   cent_array,
                                   f"{fname}_{cent_name}.pdb")


def write_matrices(centrality_dict_edge, identifiers):
    """Takes in a dictionary of dictionaries and saves a matrix
    for each inner dictionary.
    """

    for cent_name, cent_dict in centrality_dict_edge.items():
        
        # Create a graph
        G = nx.Graph()
        
        # Add nodes from the nodes' identifiers
        G.add_nodes_from(identifiers)
        
        # Create a list of (node1, node2, edge weight)
        edges = \
            [(edge[0], edge[1], cent) for edge, cent \
             in cent_dict.items()]
        
        # Add the edges to the graph
        G.add_weighted_edges_from(edges)
        
        # Convert graph to matrix
        matrix = nx.to_numpy_matrix(G)
        
        # Save the matrix with the name of the centrality measure
        np.savetxt(f"{cent_name}.dat", matrix)



######################### CENTRALITY MEASURES #########################



#-------------------------- Node centrality --------------------------#


def get_hubs(G, **kwargs):
    """Returns a dictionary mapping the nodes to their degree if
    they have a degree higher than a certain threshold (hubs),
    0 otherwise.
    """

    # Compute the degree for all nodes
    degree = G.degree()
    # Map the nodes to their degree if they are hubs, to 0 otherwise
    hubs = {n : (d if d >= kwargs["hub"] else 0) for n, d in degree}
    # Return the dictionary
    return hubs


def get_degree_centrality(G, **kwargs):
    """Returns a dictionary of degree centrality values for all nodes.
    """

    # Compute and return the degree cenntrality
    return nxc.degree_centrality(G)


def get_betweenness_centrality(G, **kwargs):
    """Returns a dictionary of betweenness centrality
    values for all nodes.
    """

    # Get the graph without glycine residues
    H = get_graph_without_glycine(G, 
                                  kwargs["identifiers"],
                                  kwargs["residue_names"])

    # Calculate the betweenness centrality values
    centrality_dict = \
        nxc.betweenness_centrality(G = H,
                                   normalized = kwargs["normalized"],
                                   weight = kwargs["weight"],
                                   endpoints = kwargs["endpoints"])
    
    # Return the finalized the dictionary of centrality values
    return finalize_dict(G, centrality_dict)


def get_closeness_centrality(G, **kwargs):
    """Returns a dictionary of closeness centrality
    values for all nodes.
    """
    
    # Get the graph without glycine residues
    H = get_graph_without_glycine(G, 
                                  kwargs["identifiers"],
                                  kwargs["residue_names"])
    
    # Calculate the closeness centrality values
    centrality_dict = \
        nxc.closeness_centrality(G = G,
                                distance = kwargs["weight"])
    
    # Return the finalized the dictionary of centrality values
    return finalize_dict(G, centrality_dict)


def get_eigenvector_centrality(G, **kwargs):
    """Returns a dictionary of eigenvector centrality
    values for all nodes.
    """

    return nxc.eigenvector_centrality_numpy(G = G, 
                                            weight = kwargs["weight"])


def get_current_flow_betweenness_centrality(G, **kwargs):
    """Returns a dictionary of current flow betweenness centrality
    values. This is the same as calculating betweenness centrality 
    using random walks instead of shortest paths.
    """

    return nxc.current_flow_betweenness_centrality(\
                G = G, 
                normalized = kwargs["normalized"],
                weight = kwargs["weight"])


def get_current_flow_closeness_centrality(G, **kwargs):
    """Returns a dictionary of current flow closeness centrality
    values.
    """

    return nxc.current_flow_closeness_centrality(\
                                G = G,
                                weight = kwargs["weight"])



#-------------------------- Edge centrality --------------------------#



def get_edge_betweenness_centrality(G, **kwargs):
    """Returns a dictionary of edge betweenness centrality values.
    """
    
    # Calculate the edge betweenness centrality values
    centrality_dict = \
        nxc.edge_betweenness_centrality(\
            G = G,
            normalized = kwargs["normalized"],
            weight = kwargs["weight"])
    
    # Reorder the edge names and return the reordered dictionary
    # of centrality values
    return reorder_edge_names(centrality_dict)


def get_edge_current_flow_betweenness_centrality(G, **kwargs):
    """Returns a dictionary of edge current flow betweenness centrality 
    values. This is the same as calculating edge betweenness centrality 
    using random walks instead of shortest paths.
    """
    
    # Calculate the edge current flow betweenness centrality values
    centrality_dict = \
        nxc.edge_current_flow_betweenness_centrality(\
            G = G,
            normalized = kwargs["normalized"],
            weight = kwargs["weight"])

    # Reorder the edge names and return the reordered dictionary
    # of centrality values
    return reorder_edge_names(centrality_dict)



# Dictionary mapping the names of the centrality measures with the
# functions calculatin them
name2function = {
    "hubs" : get_hubs,
    "degree" : get_degree_centrality, 
    "betweenness" : get_betweenness_centrality,
    "closeness" : get_closeness_centrality,
    "eigenvector" : get_eigenvector_centrality,
    "current_flow_betweenness" : \
        get_current_flow_betweenness_centrality,
    "current_flow_closeness" : \
        get_current_flow_closeness_centrality,
    "edge_betweenness" : \
        get_edge_betweenness_centrality,
    "edge_current_flow_betweenness" : \
        get_edge_current_flow_betweenness_centrality
    }



def main():



    ######################### ARGUMENT PARSER #########################



    description = \
        "Centrality analysis module for PyInteraph. It can be used " \
        "to calculate hubs, node centralities and edge centralities."
    parser = argparse.ArgumentParser(description = description)

    i_helpstr = ".dat file matrix"
    parser.add_argument("-i", "--input-dat",
                        help = i_helpstr,
                        type = str)

    r_helpstr = "Reference PDB file"
    parser.add_argument("-r", "--ref",
                        help = r_helpstr,
                        default = None,
                        type = str)

    # Node centrality types
    node = \
        ["hubs", "degree", "betweenness", "closeness", "eigenvector", 
         "current_flow_betweenness", "current_flow_closeness"]
    # Edge centrality types
    edge = ["edge_betweenness", "edge_current_flow_betweenness"]
    # All centrality types
    all_cent = node + edge

    c_choices = all_cent + ["all", "node", "edge"]
    c_helpstr = \
        f"Centrality measures to calculate. " \
        f"Selecting 'node' will calculate the following centrality " \
        f"measures {', '.join(node)}. Selecting 'edge' will calculate " \
        f"the following centrality measures : {', '.join(edge)}. " \
        f"Selecting 'all' will calculate all node and edge " \
        f"centrality measures."
    parser.add_argument("-c", "--centrality",
                        nargs = "+",
                        choices = c_choices,
                        help =  c_helpstr)

    sort_node_choices = node
    sort_node_default = "node"
    sort_node_helpstr = \
        f"Sort node centralities. Use the name of the " \
        f"desired measure. The name must match one of the " \
        f"names used in option -c. (default: {sort_node_default})."
    parser.add_argument("--sort-node",
                        choices = sort_node_choices,
                        default = sort_node_default,
                        help = sort_node_helpstr)

    sort_edge_choices = edge
    sort_edge_default = "edge"
    sort_edge_helpstr = \
        f"Sort edge centralities. Use the name of the " \
        f"desired measure. The name must match one of the " \
        f"names used in option -c. (default: {sort_edge_default})."
    parser.add_argument("--sort-edge",
                        choices = sort_edge_choices,
                        default = sort_edge_default,
                        help = sort_edge_helpstr)

    n_helpstr = f"Normalize centrality measures."
    parser.add_argument("-n", "--normalize",
                        action = "store_true",
                        help = n_helpstr)

    e_helpstr = f"Use endpoints when calculating centrality measures."
    parser.add_argument("-e", "--use-endpoints",
                        action = "store_true",
                        help = e_helpstr)

    ecx_default = 100
    ecx_helpstr = \
        f"Maximum number of iterations when calculating " \
        f"eigenvector centrality (default: {ecx_default})."
    parser.add_argument("--ec-max-iter",
                        type = int,
                        default = ecx_default,
                        help = ecx_helpstr)

    ect_default = 1e-06
    ect_helpstr = \
        f"Tolerance when calculating eigenvector " \
        f"centrality (default: {ect_default})."
    parser.add_argument("--ec-tolerance",
                        type = float,
                        default = ect_default,
                        help = ect_helpstr)

    hcut_default = 3
    hcut_helpstr = \
        f"Minimum cutoff for a node to be considered " \
        f"a hub (default: {hcut_default})."
    parser.add_argument("--hub-cutoff",
                        type = int,
                        default = hcut_default,
                        help = hcut_helpstr)

    o_default = "centrality"
    o_helpstr = \
        f"Output file for centrality measures (deafult: " \
        f"'{o_default}_node.csv' for node centralities, " \
        f"'{o_default}_edge.csv' for edge centralities)."
    parser.add_argument("-o", "--output-csv",
                        default = o_default,
                        help = o_helpstr)
    
    op_helpstr = \
        f"For each centrality measure calculated, create a PDB file " \
        f"where the bfactor column is replaced by the centrality " \
        f"value."
    parser.add_argument("-op", "--output-pdb",
                        action = "store_true",
                        help = op_helpstr)

    om_helpstr = \
        f"For each edge centrality measure calculated, create a " \
        f".dat file (matrix) containing the centrality values. "
    parser.add_argument("-om", "--output-matrix",
                        action = "store_true",
                        help = om_helpstr)

    args = parser.parse_args()
    input_dat = args.input_dat
    ref = args.ref
    centrality = args.centrality
    sort_node = args.sort_node
    sort_edge = args.sort_edge
    normalize = args.normalize
    use_endpoints = args.use_endpoints
    ec_max_iter = args.ec_max_iter
    ec_tolerance = args.ec_tolerance
    hub_cutoff = args.hub_cutoff
    output_csv = args.output_csv
    output_pdb = args.output_pdb
    output_matrix = args.output_matrix



    ########################### INPUT CHECK ###########################



    # Check that the matrix was provided
    if not input_dat:
        # Exit if the adjacency matrix was not speficied
        errstr = "Graph adjacency matrix must be specified. Exiting..."
        log.error(errstr)
        exit(1)

    # Find all centralities
    if "all" in centrality:
        centrality_names = all_cent
        
    # Find all node centralities
    elif "node" in centrality:
        centrality_names = node
        
    # Find all edge centralities
    elif "edge" in centrality:
        centrality_names = edge
        
    else:
        # Get list of specified centrality names
        centrality_names = centrality

    # Check sorting options
    if not (sort_node == "node" or sort_node in centrality_names):
        # Expected node centrality names
        expected_names = \
            [name for name in centrality_names if name not in edge]
        # Log the error
        errstr = \
            f"Option --sort-node) must be one of the following: " \
            f"{', '.join(expected_names)}. Exiting..."
        log.error(errstr)
        exit(1)

    if not (sort_edge == "edge" or sort_edge in centrality_names):
        # Expected edge centrality names
        expected_names = \
            [name for name in centrality_names if name not in edge]
        # Log the error
        errstr = \
            f"Option --sort-edge must be one of the following: " \
            f"{', '.join(expected_names)}. Exiting..."
        log.error(errstr)
        exit(1)



    ######################## GRAPH & NODE NAMES #######################



    # Build the graph and get identifiers and residue names for
    # the nodes of the graph from the reference structure
    identifiers, residue_names, graph = \
        pa.build_graph(input_dat, ref)

    # Get graph nodes and edges
    nodes = graph.nodes()
    edges = graph.edges()
    
    # Log info about the graph
    identifiers_str = "\n".join(identifiers)
    infostr = \
        f"Graph loaded! {len(nodes)} nodes, {len(edges)} edges\n" \
        f"Nodes: {identifiers_str}\n"
    log.info(infostr)



    ############################ CENTRALITY ############################



    # Create dictionary of optional arguments
    kwargs = {"weight" : None,
              "normalized" : normalize,
              "endpoints" : use_endpoints,
              "max_iter" : ec_max_iter,
              "tol" : ec_tolerance,
              "hub" : hub_cutoff,
              "identifiers" : identifiers,
              "residue_names" : residue_names}
        
    # Get dictionary of node and edge centrality values
    node_dict, edge_dict = \
        get_centrality_dict(centrality_list = centrality_names,
                            name2function = name2function, 
                            graph = graph,
                            **kwargs)

    # If the dictionary is not empty, node centralities
    # have been requested
    if len(node_dict) > 1:
            
        # Save the dictionary as a table
        write_table(fname = f"{output_csv}_node.csv",
                    centrality_dict = node_dict,
                    sort_by = sort_node)

        # Write the PDB files if requested
        if output_pdb and ref is not None:
            write_pdb_files(centrality_dict_node = node_dict,
                            pdb = ref,
                            fname = output_pdb)

    # If the dictionary is not empty, edge centralities
    # have been requested
    if edge_dict:

        # Save the dictionary as a table            
        write_table(fname = f"{output_csv}_edge.csv",
                    centrality_dict = edge_dict, 
                    sort_by = sort_edge)

        # Write the matrices if requested
        if output_matrix and ref is not None:
            write_matrices(centrality_dict_edge = edge_dict, 
                           identifiers = identifiers,
                           fname = output_matrix)

if __name__ == "__main__":
    main()
