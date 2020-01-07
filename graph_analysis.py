#!/usr/bin/python
# -*- coding: utf-8 -*-

#    PyInteraph, a software suite to analyze interactions and interaction network in structural ensembles.
#    Copyright (C) 2013 Matteo Tiberti <matteo.tiberti@gmail.com>, Gaetano Invernizzi, Yuval Inbar,
#    Matteo Lambrughi, Gideon Schreiber,  Elena Papaleo <elena.papaleo@unimib.it> <elena.papaleo@bio.ku.dk>
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.

#   This program is distributed in the hope that it will be useful,
#   but WITHOUT ANY WARRANTY; without even the implied warranty of
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#   GNU General Public License for more details.

#   You should have received a copy of the GNU General Public License
#   along with this program.  If not, see <http://www.gnu.org/licenses/>.

from __future__ import absolute_import
import sys
import logging as log
import argparse

vinfo = sys.version_info
if vinfo[0] < 2 or (vinfo[0] == 2 and vinfo[1] < 7):
    errstr = \
        "Your Python version is {:s}, but only " \
        "versions >= 2.7 are supported."
    log.error(errstr.format(sys.version))
    exit(1)

import numpy as np
import networkx as nx
import MDAnalysis as mda
from Bio import PDB

########################### ARGUMENT PARSER ###########################

description = "PyInteraph network analysis module."
parser = argparse.ArgumentParser(description = description)

r_helpstr = "Reference topology file"
parser.add_argument("-r", "--reference", \
                    metavar = "TOPOLOGY", \
                    dest = "top", \
                    type = str, \
                    default = None, \
                    help = r_helpstr)

a_helpstr = "Input graph file"
parser.add_argument("-a", "--adj-matrix", \
                    metavar = "DAT", \
                    dest = "dat", \
                    type = str, \
                    default = None, \
                    help = a_helpstr)

c_helpstr = "Calculate connected components"
parser.add_argument("-c", "--components", \
                    dest = "do_components", \
                    action = "store_true", \
                    default = False, \
                    help = c_helpstr)

u_helpstr = "Calculate hubs"
parser.add_argument("-u", "--hubs", \
                    dest = "do_hubs", \
                    action = "store_true", \
                    default = False, \
                    help = u_helpstr)

k_default = 3
k_helpstr = "Minimum number of connections for hubs (default: {:d})"
parser.add_argument("-k", "--hubs-cutoff", \
                    dest = "hubs_cutoff", \
                    default = k_default, \
                    type = int, \
                    help = k_helpstr.format(k_default))

p_helpstr = "Calculate all simple paths between " \
            "two residues in the graph"
parser.add_argument("-p", "--all-paths", \
                    dest = "do_paths", \
                    action = "store_true", \
                    default = False, \
                    help = p_helpstr)

r1_helpstr = "First residue for paths calculation (see option -p)"
parser.add_argument("-r1", "--source", \
                    dest = "source", \
                    default = None, \
                    help = r1_helpstr)

r2_helpstr = "Last residue for paths calculation (see option -p)"
parser.add_argument("-r2", "--target", \
                    dest = "target", \
                    default = None, \
                    help = r2_helpstr)

l_helpstr = "Maximum path length (see option -p) (default: {:d})"
parser.add_argument("-l", "--maximum-path-length", \
                    dest = "maxl", \
                    default = 10, \
                    type = int, \
                    help = )

s_choices = ["length", "cumulative_weight", "avg_weight"]
s_default = "lenght"
s_helpstr = \
    "How to sort pathways in output. Possible choices are {:s}" \
    " (default: {:s})"
parser.add_argument("-s", "--sort-paths", \
                    dest = "sort_paths_by", \
                    choices = s_choices, \
                    default = "length", \
                    help = \
                        s_helpstr.format(", ".join(s_choices), s_default))

cb_helpstr = "Save connected components ID in PDB file"
parser.add_argument("-cb", "--components-pdb", \
                    dest = "components_pdb", \
                    default = None, \
                    help = cb_helpstr)

ub_helpstr = "Save hub degrees in PDB file"
parser.add_argument("-ub", "--hubs-pdb", \
                    dest = "hubs_pdb", \
                    default = None, \
                    help = cb_helpstr)

d_helpstr = "Write the paths found as matrices"
parser.add_argument("-d", "--write-paths", \
                    dest = "write_paths", \
                    default = False, \
                    action = "store_true", \
                    help = d_helpstr)

args = parser.parse_args()


############################## FUNCTIONS ##############################

def residue_key(x):
    return int("".join(list(filter(str.isdigit, str(x)))))


def values_to_bfac(pdb, vals, pdb_out):
    parser = PDB.PDBParser()
    structure = parser.get_structure("protein", pdb)
    io = PDB.PDBIO()
    chain_offset = 0
    for model in structure:
        for chain in model:
            for i, residue in enumerate(chain):
                for atom in residue:
                    atom.set_bfactor(float(vals[i+chain_offset]))
            chain_offset += len(chain)

    io.set_structure(structure)
    io.save(pdb_out)


def build_graph(fname, pdb = None):
    try:
        data = np.loadtxt(fname)
    except:
        errstr = \
            "Could not load file {:s} or wrong file format. Exiting..."
        # log the full numpy stack trace with exc_info
        log.error(errstr.format(fname), exc_info = True)
        exit(1)

    if pdb is not None:
        try:
            # generate a Universe object from the PDB file
            # (use the PDB for both the 'topology' and 'trajectory'
            # arguments necessary to create the Universe)
            u = mda.Universe(pdb, pdb)
        except Exception:
            errstr = \
                "Exception caught during creation of the Universe " \
                "object. Exiting..."
            # log the full MDAnalysis stack trace with exc_info
            log.error(errstr, exc_info = True)
            exit(1)
        
        identifiers = \
            ["{:s}-{:d}{:s}".format(r.segment.segid, \
                                    r.resnum, \
                                    r.resname) \
             for r in u.residues]
 
    else:
        # generate automatic identifiers going from 1 to the
        # total number of residues considered
        identifiers = [str(name) for name in range(1,data.shape[0]+1)]
    
    # generate a graph from the data loaded
    G = nx.Graph(data)

    # set the names of the graph nodes (in place)
    node_names = dict(zip(range(data.shape[0]), identifiers))
    nx.relabel_nodes(G, mapping = node_names, copy = False)

    return identifiers, G


def get_connected_components(G):
    return list(nx.connected_components(G))


def write_connected_components(ccs, outfile = None):
    
    # outfile = None makes it output to the console
    # (PyInteraph 1 behavior)

    logstr = "Connected component {:d}\n ({:d} elements) {:s}"

    if outfile is None:
        for i, cc in enumerate(ccs):
            ### TODO: output to a file, not print
            print(logstr.format(\
                    i + 1, \
                    len(cc), \
                    ", ".join(sorted(cc, key = residue_key))))

    else:
        raise NotImplementedError


def write_connected_components_pdb(identifiers, \
                                   ccs, \
                                   top, \
                                   components_pdb, \
                                   conversion_func):
    
    conn_comp_array = np.array(identifiers)
    for i, cc in enumerate(ccs):
        for res in cc:
            conn_comp_array[identifiers.index(res)] = i+1
    # write a PDB file identical to the reference PDB file
    # but with the b-factor column filled with the number of
    # the connected component a residue belongs to
    conversion_func(pdb = top, \
                    vals = conn_comp_array, \
                    pdb_out = components_pdb)


def get_hubs(G, min_k = 3, sorting = None):
    hubs = {node : k for node, k in G.degree() if k >= min_k}
    
    if len(list(hubs.keys())) == 0:
        # warn the user that no hubs have been found
        logstr = \
            "No hubs with connectivity >= {:d} have been found."
        log.warning(logstr.format(min_k))
        
        return hubs
    
    else:
        if sorting is None:
            sorted_hubs = hubs
        elif sorting == "ascending":
            sorted_hubs = \
                sorted(hubs.items(), \
                       key = lambda item: item[1], \
                       reverse = False)
        elif sorting == "descending":
                sorted(hubs.items(), \
                       key = lambda item: item[1], \
                       reverse = True)

        return dict(sorted_hubs)


def write_hubs(hubs, outfile = None):

    if outfile is None:
        ### TODO: output to a file, not print
        print("Hubs:\n\tNode\tk{:s}")
        for hub, k in hubs:
            print("\t{:s}\t{:d}".format(hub, k))

    else:
        raise NotImplementedError


def write_hubs_pdb(identifiers, \
                   hubs, \
                   top, \
                   hubs_pdb, \
                   conversion_func):
    
    hubs_array = np.zeros(len(identifiers))
    for index, node in enumerate(identifiers):
        if node in set(hubs.keys()):
            # check if the node is a hub
            # iterate and check over a set()
            # of keys because it is faster than
            # a list (keys() returns a list
            # in python2)
            hubs_array[index] = hubs[node]
    # write a PDB file identical to the reference PDB file
    # but with the b-factor column filled with the degree of
    # a residue if it is a hub, zero otherwise       
    conversion_func(pdb = args.top, \
                    vals = hubs_array, \
                    pdb_out = args.hubs_pdb) 


def get_paths(G, source, target, maxl, sort_paths_by):
    if source is None or target is None:
        log.error(\
            "You must specify source and target residues. Exiting...")
        exit(1)
        
    if not source in G.nodes() or not target in G.nodes():
        # maybe logstring should reflect better the error?
        log.error(\
            "Source or target residues have been badly specified. " \
            "Exiting...")
        exit(1)
        
    try:
        shortest = nx.algorithms.shortest_path(G = G, \
                                                source = source, \
                                                target = target)
    except nx.NetworkXNoPath:
        log.warning("No paths exist between selected residues.")
        exit(1)

    if len(shortest) > maxl:
        warnstr = \
            "No paths were found between the given nodes at the " \
            "given cut-off. Shortest path length between these two " \
            "nodes is {:d}"
        log.warning(warnstr.format(len(shortest)))
        exit(1)
        
    else:
        # calculate all the paths
        paths = \
            list(nx.algorithms.simple_paths.all_simple_paths(\
                                                    G = G, \
                                                    source = source, \
                                                    target = target, \
                                                    cutoff = maxl))

        lengths = [len(p) for p in paths]
            
        # calculate the sum of their weights
        sum_weights = \
            [np.sum([G[p[i]][p[i+1]]["weight"] \
                    for i in range(len(p)-1)]) \
            for p in paths]
            
        # calculate the average of their weights
        avg_weights = \
            [np.average([G[p[i]][p[i+1]]["weight"] \
                for i in range(len(p)-1)]) \
            for p in paths]

        # sort the paths
        full_paths = zip(paths, lengths, sum_weights, avg_weights)        
        reverse = True        
        if sort_paths_by == "length":
            key = lambda x: x[1]
            reverse = False            
        elif sort_paths_by == "cumulative_weight":
            key = lambda x: x[2]
        elif sort_paths_by == "avg_weight":
            key = lambda x: x[3]

        sorted_paths = sorted(full_paths, \
                              key = key, \
                              reverse = reverse)

        return sorted_paths


def write_paths(paths, outfile = None)

    # write the paths found to the output in a human-readable format
    if outfile is None:
        print("Path #\tLength\tSum of weights\tAverage weight\tPath")
        pathfmt_str = "{:d}\t{:d}\t{:.1f}\t\t{:.1f}\t\t{:s}"
        for index, path in enumerate(paths):
            print(\
                pathfmt_str.format(\
                    index+1, path[1], path[2], path[3], ",".join(path[0])))

    else:
        raise NotImplementedError


def write_paths_matrices(identifiers, G, paths, fmt):
    for index, path in enumerate(paths):
        # for each path...
        path_matrix = np.zeros(nx.adjacency_matrix(G).shape)
        for node in range(len(path[0])-1):
            # for each node in the path...
            x = identifiers.index(path[0][node])
            y = identifiers.index(path[0][node+1])
            path_matrix[x,y] = \
                G[path[0][node]][path[0][node+1]]["weight"]
            path_matrix[y,x] = \
                G[path[0][node+1]][path[0][node]]["weight"]
        # save the matrix
        matrix_file = "path{:s}.dat".format(index+1)
        np.savetxt(matrix_file, path_matrix, fmt = fmt)
    


if __name__ == "__main__":

    ############################## MAIN ###############################

    if args.dat is None:
        # Exit if the adjacency matrix was not speficied
        log.error("Graph adjacency matrix must be specified. Exiting...")
        exit(1)

    if (args.components_pdb is None or args.hubs_pdb is None) \
    and args.top is None:
        # Exit if the user requested the PDB files with connected
        # components and hubs but did not provide a reference PDB
        # file
        log.error(\
            "A PDB reference file must be supplied with options " \
            "-cb and -ub. Exiting...")
        exit(1)

    # build the graph
    identifiers, G = build_graph(args.dat, pdb = args.top)
    nodes = G.nodes()
    edges = G.edges()

    logstr = "Graph loaded! {:d} nodes, {:d} edges"
    log.info(logstr.format(len(nodes), len(edges)))
    logstr = "Node list: {:s}"
    log.info(\
        logstr.format(\
            "\t".join([node for node in sorted(nodes, key = residue_key)])))


    ###################### CONNECTED COMPONENTS #######################

    if args.do_components:
        # calculate the connected components
        ccs = get_connected_components(G)
        # write the connected components
        write_connected_components(ccs = ccs, outfile = None)

        if args.components_pdb is not None:
            # write PDB file with B-factor column replaced
            write_connected_components_pdb(\
                identifiers = identifiers, \
                ccs = ccs, \
                top = args.top, \
                components_pdb = args.components_pdb, \
                conversion_func = values_to_bfac):


    ############################## HUBS ###############################
            
    if args.do_hubs:
        # calculate the hubs
        hubs = get_hubs(G = G, \
                        min_k = args.hubs_cutoff, \
                        sorting = "descending")
        # write the hubs
        write_hubs(hubs = hubs, outfile = None)
        
        if args.hubs_pdb is not None:
            # write PDB file with B-factor column replaced
            write_hubs_pdb(\
                identifiers = identifiers, \
                hubs = hubs, \
                top = args.top, \
                hubs_pdb = args.hubs_pdb, \
                conversion_func = values_to_bfac)


    ############################## PATHS ##############################
            
    if args.do_paths:
        # calculate paths between a pair of residues
        paths = get_paths(G = G, \
                          source = args.source, \
                          target = args.target, \
                          max = args.maxl, \
                          sort_paths_by = args.sort_paths_by)
        # write the paths
        write_paths(paths = paths, outfile = None)
            
        if args.write_paths:
            # write paths as matrices
            write_paths_matrices(identifiers = identifiers, \
                                 G = G, \
                                 paths = paths, \
                                 fmt = "%.1f")

