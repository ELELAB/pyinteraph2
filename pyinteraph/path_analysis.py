#!/usr/bin/env python

#   path_analysis.py: script to calculate paths and path-related
#                     network properties
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

import os
import sys
import argparse
import logging as log
import numpy as np
import networkx as nx
import MDAnalysis as mda
import itertools
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib

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
        residue_names = [r.resname for r in u.residues]
    # if the user did not provide a reference structure
    else:
        # generate automatic identifiers going from 1 to the
        # total number of residues considered
        identifiers = [f"_{i}" for i in range(1, adj_matrix.shape[0]+1)]
        residue_names = None
    
    # generate a graph from the data loaded
    G = nx.Graph(adj_matrix)
    # set the names of the graph nodes (in place)
    node_names = dict(zip(range(adj_matrix.shape[0]), identifiers))
    nx.relabel_nodes(G, mapping = node_names, copy = False)
    # return the idenfiers and the graph
    return identifiers, residue_names, G

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


def get_persistence_graph(graph, paths, identifiers):
    """Takes in a PSN graph, a list of paths and a list of nodes in the 
    PSN and returns a new graph of paths where each edge corresponds to 
    persistence value of the original PSN. All nodes from the PSN are
    also included in the output graph.
    """

    pers_graph = nx.Graph()
    # Add all nodes
    pers_graph.add_nodes_from(identifiers)
    # Add all paths
    for p in paths:
        nx.add_path(pers_graph, p)
    # Add all edge weights
    edge_weights = [(u, v, graph[u][v]['weight']) for u, v in pers_graph.edges()]
    pers_graph.add_weighted_edges_from(edge_weights)
    return pers_graph

def get_shortest_paths(graph, source, target):
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
                path = list(nx.algorithms.shortest_paths.generic.all_shortest_paths(
                                G = graph,
                                source = node1,
                                target = node2))
                for p in path:
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
        path = list(nx.algorithms.simple_paths.all_simple_paths(G = graph, 
                                                                source = node1,
                                                                target = node2, 
                                                                cutoff = maxl))
        # Only add paths to output if they exist
        for p in path:
            if len(p) > 0:
                paths.append(p)
    return paths

def sort_paths(graph, paths, sort_by):
    """Takes in a list of paths and returns a table of sorted paths"""
    
    # Get source and target
    source = [p[0] for p in paths]
    target = [p[-1] for p in paths]
    # Calculate length of path
    lengths = [len(p) for p in paths]
    # Calculate weights of path
    weights = [[graph[p[i]][p[i+1]]["weight"] for i in range(len(p)-1)] \
                for p in paths]
    sum_weights = [np.sum(w) for w in weights]
    avg_weights = [np.mean(w) for w in weights]
    # Sort paths
    paths = zip(paths, source, target, lengths, sum_weights, avg_weights)
    if sort_by == "path":
        key = lambda x: x[0]
        reverse = False
    elif sort_by == "length":
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

def write_table(fname, table):
    """Save sorted table as txt file. """

    # Remove any file extensions
    fname = os.path.splitext(fname)[0]
    # For each line in table, add line in file
    with open(f"{fname}.txt", "w") as f:
        # Write the table header
        f.write("path\tsource\ttarget\tlength\tsum_weights\tavg_weight\n")
        # For each path found
        for path, source, target, length, sum_weights, avg_weight in table:
            path = ','.join(path)
            line = f"{path}\t{source}\t{target}\t{length}" \
                   f"\t{sum_weights}\t{avg_weight}\n"
            f.write(line)

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
                    if abs(idx1 - idx2) >= res_space \
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
                            G = graph,
                            source = node1,
                            target = node2))
            for p in path:
                # Add path to output if it contains more than one edge
                if len(p) > 2:
                    paths.append(p)
        except Exception:
            # If no path is found log info
            log.debug(f"No path found between {node1} and {node2}")
    return paths

def get_graph_from_paths(paths):
    """Takes in a list of paths and returns a corresponding graph where
    the weight of each edge is its frequency and the weight of each node
    is its frequency.
    """

    # Increment amount
    inc = 1/len(paths)
    # Build graph
    graph = nx.Graph()
    for path in paths:
        for i in range(len(path) - 1):
            # Get edge
            node1 = path[i]
            node2 = path[i+1]
            # If the node already exists
            if graph.has_edge(node1, node2):
                # Always add first node
                if i == 0:
                    graph.nodes()[node1]["n_weight"] += inc
                # Increment second node weight
                graph.nodes()[node2]["n_weight"] += inc
                # Increment edge weight
                graph[node1][node2]["e_weight"] += inc
            # If edge does not exist
            else:
                # One of the nodes may exist, always check first node
                if i == 0:
                    # Increment node if present, else add
                    if node1 in graph:
                        graph.nodes()[node1]["n_weight"] += inc
                    else:
                        graph.add_node(node1, n_weight = inc)
                # Increment second node if present, else add
                if node2 in graph:
                    graph.nodes()[node2]["n_weight"] += inc
                else:
                    graph.add_node(node2, n_weight = inc)
                graph.add_edge(node1, node2, e_weight = inc)
    return graph

def filter_graph(graph, node_threshold, edge_threshold):
    """Takes in a graph, and returns a graph which only contains nodes
    and edges from the provided graph that have a weight higher than
    the provided threshold.
    """

    filterd_graph = nx.Graph()
    for u, v, d in graph.edges(data = True):
        # Check that both thresholds are met
        if graph.nodes()[u]["n_weight"] > node_threshold and \
           graph.nodes()[v]["n_weight"] > node_threshold and \
           d["e_weight"] > edge_threshold:
            # Add node weights first
            filterd_graph.add_node(u, n_weight = graph.nodes()[u]["n_weight"])
            filterd_graph.add_node(v, n_weight = graph.nodes()[v]["n_weight"])
            # Add edge weight
            filterd_graph.add_edge(u, v, e_weight = d["e_weight"])
    # Warn if no edges in metapath graph
    if nx.is_empty(filterd_graph):
        log.warning("No metapaths found.")
    return filterd_graph

def normalize_graph(graph):
    """Takes in a graph where all weights ranged between 0 to 1 and 
    returns a graph where all weights are normalized (divided by max
    weight)
    """

    normalized_graph = nx.Graph()
    # Get max weights for nodes and edges
    max_edge = max([d["e_weight"] for u, v, d in graph.edges(data = True)])
    max_node = max([d["n_weight"] for n, d in graph.nodes(data=True)])
    
    for u, v, d in graph.edges(data = True):
        # Add nodes first
        normalized_graph.add_node(u, n_weight = \
                                    graph.nodes()[u]["n_weight"]/max_node)
        normalized_graph.add_node(v, n_weight = \
                                    graph.nodes()[v]["n_weight"]/max_node)
        # Add edge
        normalized_graph.add_edge(u, v, e_weight = d["e_weight"]/max_edge)

    return normalized_graph

def get_metapath(graph, res_id, res_space, node_threshold, edge_threshold, normalize):
    """Takes in a PSN graph where weights are persistence values. Returns
    a graph of the metapath
    """
    
    # Calculate all shortest paths
    paths = get_all_shortest_paths(graph, res_id, res_space)
    # Create graph from path list
    paths_graph = get_graph_from_paths(paths)
    # Normalize graph
    if normalize:
        paths_graph = normalize_graph(paths_graph)
    # Filter graph
    metapath_graph = filter_graph(paths_graph, node_threshold, edge_threshold)
    return metapath_graph

def plot_graph(fname, graph, hub_num, col_map_e, col_map_n, dpi, node_space, node_size, patch_col):
    """Takes in a graph and saves a pdf of the plot. Also takes in a hub
    cutoff value. Nodes with a larger number of edges than the cutoff
    are highlighted.
    """

    # Get attributes
    weights = [d["e_weight"] for u, v, d in graph.edges(data=True)]
    nodes = np.array([n for n, d in graph.degree(graph.nodes())])
    degrees = np.array([d for n, d in graph.degree(graph.nodes())])
    selection = degrees >= hub_num
    hubs = nodes[selection]
    hubs_deg = degrees[selection]
    non_hubs = nodes[np.logical_not(selection)]
    non_hubs_deg = degrees[np.logical_not(selection)]
    unique_hubs = len(np.unique(hubs_deg))
    # Get positions. Larger k values make the nodes spread out more
    pos = nx.spring_layout(graph, k=node_space, iterations=30)
    # Get cmap for edges
    edge_colors = sns.color_palette(palette = col_map_e, n_colors =  256)
    cmap_e = ListedColormap(edge_colors, name = 'edge_colors', N = 256)
    # Remove border
    _, ax = plt.subplots()
    ax.axis('off')
    # Draw non hubs
    nx.draw_networkx_nodes(graph, 
                           pos, 
                           nodelist = list(non_hubs),
                           node_size = node_size,
                           node_color = 'white',
                           edgecolors = 'black',
                           label = list(non_hubs_deg))
    # Draw hubs
    if unique_hubs == 1:
        # Color the unique hub gray
        nx.draw_networkx_nodes(graph, 
                               pos, 
                               nodelist = list(hubs),
                               node_size = node_size,
                               node_color = 'gray',
                               edgecolors = 'black')
        # Add label for single unique hub
        label = matplotlib.patches.Patch(color = patch_col, 
                                         label = np.unique(hubs_deg)[0])
        plt.legend(handles=[label], title="Node degree")

    # If more than one unique hub, add colorbar
    if unique_hubs > 1:
        # Use colormap if more than 1 unique hubs
        # Gray scale for nodes (select how many greys to pick)
        node_colors = sns.color_palette(palette = col_map_n, 
                                        n_colors = unique_hubs)

        cmap_n = LinearSegmentedColormap.from_list(name = 'node_colors', 
                                                   colors = node_colors, 
                                                   N = len(node_colors))
        nx.draw_networkx_nodes(graph, 
                               pos, 
                               nodelist = list(hubs),
                               node_size = node_size,
                               node_color = list(hubs_deg),
                               cmap = cmap_n,
                               edgecolors = 'black',
                               label = list(hubs_deg))
    # Draw edges
    nx.draw_networkx_edges(graph, 
                           pos, 
                           edge_color = weights, 
                           edge_cmap = cmap_e,
                           edge_vmin = 0, 
                           edge_vmax= 1,
                           width = 3)
    # Draw labels
    nx.draw_networkx_labels(graph, pos, font_size=9, font_color ='black')
    # Add color bar
    # Resize colorbar
    divider = make_axes_locatable(ax)
    cax_e = divider.append_axes("right", size="5%", pad=0.2)
    # Edge color bar
    sm_e = plt.cm.ScalarMappable(cmap=cmap_e, norm=plt.Normalize(vmin=0, 
                                                                 vmax=1))
    cbar_e = plt.colorbar(sm_e, cax_e)
    cbar_e.set_label('Relative recurrence')
    # Node color bar (gray scale)
    if unique_hubs > 1:
        # colorbar start pos and end pos
        # extend past actual range by 0.5 to center ticks
        start = min(hubs_deg) - 0.5
        end = max(hubs_deg) + 0.5
        # shift ticks by 0.5 so they align
        tick_pos = np.arange(start, end) + 0.5
        # Colormap position
        cax_n = divider.append_axes("bottom", size="5%", pad=0.2)
        # Scalar mapper oject
        sm_n = plt.cm.ScalarMappable(cmap=cmap_n, 
                                     norm=plt.Normalize(vmin=start, 
                                                        vmax=end))
        cbar_n = plt.colorbar(sm_n, 
                              cax_n, 
                              orientation = 'horizontal',
                              ticks = tick_pos)
        # Add tick labels
        #cbar_n.ax.set_xticklabels(range(lo, hi + 1))
        cbar_n.set_label('Node Degree')
    # Save figure
    plt.savefig(f"{fname}.pdf", dpi = dpi, bbox_inches = 'tight', format = 'pdf')

def reorder_graph(graph, identifiers):
    ordered_graph = nx.Graph()
    ordered_graph.add_nodes_from(identifiers)
    weighted_edges = [(u, v, d["e_weight"]) for u, v, d in graph.edges(data=True)]
    ordered_graph.add_weighted_edges_from(weighted_edges)
    return(ordered_graph)


def main():

    ######################### ARGUMENT PARSER #########################
    
    description = "Path analysis module for PyInteraph. It allows calculation of " \
                  "shortest or simple paths between a given source and target nodes" \
                  "(see option -l) and the metapath (option -m)."
    parser = argparse.ArgumentParser(description = description)

    i_helpstr = ".dat file matrix."
    parser.add_argument("-i", "--input-dat",
                        dest = "input_matrix",
                        help = i_helpstr,
                        type = str)
    
    r_helpstr = "Reference PDB file."
    parser.add_argument("-r", "--pdb",
                        dest = "pdb",
                        help = r_helpstr,
                        default = None,
                        type = str)

    # Path analysis
    p_helpstr = "Calculate shortest or all paths between source and target "\
                "(see option -l)."
    parser.add_argument("-p", "--path",
                        dest = "do_path",
                        action = "store_true",
                        default = False,
                        help = p_helpstr)
    
    l_default = "shortest"
    l_helpstr = f"Type of path to be calculated between source and target. " \
                f"If 'shortest' is used, calculates the shortest paths between " \
                f"the source and target. If an integer is used, calculates " \
                f"all paths with a length smaller than or equal to the integer " \
                f"between the source and target (default: {l_default})."
    parser.add_argument("-l", "--path_length", 
                        dest = "path_l",
                        default = l_default,
                        type = str,
                        help = l_helpstr)

    s_helpstr = "Source residues for paths calculation (see option -l)."
    parser.add_argument("-s", "--source",
                        dest = "source",
                        default = None,
                        type = str,
                        help = s_helpstr)

    t_helpstr = "Target residues for paths calculation (see option -l)."
    parser.add_argument("-t", "--target",
                        dest = "target",
                        default = None,
                        type = str,
                        help = t_helpstr)

    b_choices = ["path", "length", "cumulative_weight", "average_weight"]
    b_default = "path"
    b_helpstr = "How to sort pathways in output. Possible choices are: " \
                f"{b_choices} (default: {b_default})."
    parser.add_argument("-b", "--sort-paths",
                        dest = "sort_by",
                        choices = b_choices,
                        default = b_default,
                        help =  b_helpstr)

    a_default = "shortest_paths"
    a_helpstr = f"Output file name for paths calculation (see option -l) " \
                f"(default: {a_default}.txt)."
    parser.add_argument("-a", "--path-output",
                        dest = "p_out",
                        default = a_default,
                        help = a_helpstr)

    # Metapath
    m_helpstr = "Calculate metapath."
    parser.add_argument("-m", "--metapath",
                        dest = "do_metapath",
                        action = "store_true",
                        default = False,
                        help = m_helpstr)

    w_helpstr = f"During metapath calculation, normalize the edge and node " \
                f"weights."
    parser.add_argument("-w", "--normalize-weights",
                        dest = "do_normalize",
                        action = "store_true",
                        default = False,
                        help = w_helpstr)

    g_default  = 2
    g_helpstr = f"During metapath calculation, only calculate paths between " \
                f"residues that are separated by this number of residues or " \
                f"more in the protein backbone. The last residue of a chain " \
                f"and the first residue of the subsequent chain are counted " \
                f"as adjacent residues (default: {g_default})."
    parser.add_argument("-g", "--residue-gap", 
                        dest = "res_gap",
                        default = g_default,
                        type = int,
                        help = g_helpstr)

    e_default  = 0.1
    e_helpstr = f"During metapath filtering, only keep edges that occur more " \
                f"frequently than this value (default: {e_default})."
    parser.add_argument("-e", "--edge-threshold", 
                        dest = "edge_thresh",
                        default = e_default,
                        type = float,
                        help = e_helpstr)

    n_default  = 0.1
    n_helpstr = f"During metapath filtering, only keep nodes that occur more " \
                f"frequently than this value (default: {n_default})."
    parser.add_argument("-n", "--node-threshold", 
                        dest = "node_thresh",
                        default = n_default,
                        type = float,
                        help = n_helpstr)

    c_default = 3
    c_helpstr = f"During metapath plotting, nodes with a degree higher than " \
                f"this value are designated as hubs (default: {c_default})."
    parser.add_argument("-c", "--hub-cutoff",
                        dest = "hub",
                        default = c_default,
                        type = int,
                        help = c_helpstr)

    k_default = 0.2
    k_helpstr = f"Node spacing during metapath plotting. Higher values cause " \
                f"nodes to be more spread apart (default: {k_default})."
    parser.add_argument("-k", "--node-spacing",
                        dest = "node_space",
                        default = k_default,
                        type = float,
                        help = k_helpstr)

    z_default = 900
    z_helpstr = f"Node size during metapath plotting. Higher values create " \
                f"larger nodes (default: {z_default})."
    parser.add_argument("-z", "--node-size",
                        dest = "node_size",
                        default = z_default,
                        type = int,
                        help = z_helpstr)

    d_default = "metapath"
    d_helpstr = f"Metapath plot name (default: {d_default}.pdf)."
    parser.add_argument("-d", "--metapath-output",
                        dest = "m_out",
                        default = d_default,
                        help = d_helpstr)
    

    args = parser.parse_args()
    

    # Check user input
    if not args.input_matrix:
        # exit if the adjacency matrix was not speficied
        log.error("Graph adjacency matrix must be specified. Exiting ...")
        exit(1)
    
    # Load file, build graphs and get identifiers for graph nodes
    identifiers, residue_names, graph = build_graph(fname = args.input_matrix,
                                                    pdb = args.pdb)

    # get graph nodes and edges
    nodes = graph.nodes()
    edges = graph.edges()
    # print nodes
    sys.stdout.write("Node list:\n")
    for node in identifiers:
        sys.stdout.write(f"{node}\n")
    sys.stdout.write(f"Graph loaded! {len(nodes)} nodes, {len(edges)} edges\n")
    if args.pdb is None:
        log.warning("No reference PDB file provided.")


    ############################## PATHS ##############################


    if args.do_path:
        # Check if source and target are provided
        if args.source and args.target:
            # Convert user input to a list of nodes
            source_list = convert_input_to_list(user_input = args.source,
                                                identifiers = identifiers)
            target_list = convert_input_to_list(user_input = args.target,
                                                identifiers = identifiers)
        else:
            log.error("Source and target must be provided when calculating paths.")
            exit(1)

        # Choost path type
        if args.path_l == "shortest":
            all_paths = get_shortest_paths(graph = graph,
                                           source = source_list,
                                           target = target_list)
        else:
            # Check integer is provided
            try:
                maxl = int(args.path_l)
            except Exception:
                errstr = "Integer must be specified for calculation of all " \
                         "simple paths. Otherwise use 'shortest'."
                raise ValueError(errstr)
            # Calculate all simple paths:
            all_paths = get_all_simple_paths(graph = graph,
                                             source = source_list,
                                             target = target_list,
                                             maxl = maxl)
            # Change output file name if user forgets to change
            if args.p_out == "shortest_paths":
                args.p_out = f"all_paths_{maxl}" 
        
        # Get persistence graph
        pers_graph = get_persistence_graph(graph = graph, 
                                           paths = all_paths, 
                                           identifiers = identifiers)
        # Create sorted table from paths
        all_paths_table = sort_paths(graph = graph,
                                     paths = all_paths,
                                     sort_by = args.sort_by)

        # Save all/shortest path table
        sys.stdout.write(f"Saving output: {args.p_out}\n")
        write_table(f"{args.p_out}", all_paths_table)

        # Save all/shortest path matrix
        path_matrix = nx.to_numpy_matrix(pers_graph)
        np.savetxt(f"{args.p_out}.dat", path_matrix)
    
    
    ############################# METAPATH ############################

    if args.do_metapath:
        if args.res_gap <= 0:
            err_str = "Residue spacing (option -k, --node-spacing) " \
                      "must be a positive integer. Exiting..."
            log.error(err_str)
            exit(1)

        # Get metapath graph
        metapath_graph = get_metapath(graph = graph,
                                      res_id = identifiers,
                                      res_space = args.res_gap,
                                      node_threshold = args.node_thresh,
                                      edge_threshold = args.edge_thresh,
                                      normalize = args.do_normalize)

        # If metapath found
        if not nx.is_empty(metapath_graph):
            # Remove file extensions
            args.m_out = os.path.splitext(args.m_out)[0]
            # Plot graph
            plot_graph(fname = f"{args.m_out}", 
                       graph = metapath_graph, 
                       hub_num = args.hub,
                       col_map_e = "rocket_r",
                       col_map_n = "gray_r",
                       dpi = 100,
                       node_space = args.node_space,
                       node_size = args.node_size,
                       patch_col = 'gray')

            # Fill metapath graph with nodes for all residues
            metapath_graph = reorder_graph(metapath_graph, identifiers)
            # Create matrix
            sys.stdout.write(f"Saving output: {args.m_out}\n")
            metapath_matrix = nx.to_numpy_matrix(metapath_graph)
            np.savetxt(f"{args.m_out}.dat", metapath_matrix)


if __name__ == "__main__":
    main()
