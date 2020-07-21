#!/usr/bin/env python
# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-

#    PyInteraph, a software suite to analyze interactions and 
#    interaction network in structural ensembles.
#    Copyright (C) 2013 Matteo Tiberti <matteo.tiberti@gmail.com>, 
#                       Gaetano Invernizzi, Yuval Inbar, 
#                       Matteo Lambrughi, Gideon Schreiber, 
#                       Elena Papaleo <elena.papaleo@unimib.it> 
#                                     <elena.papaleo@bio.ku.dk>
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

import argparse
import logging as log
import os
import os.path
import sys

import MDAnalysis as mda
import numpy as np
from libinteract import libinteract as li


######################## ENVIRONMENT VARIABLES ########################

INSTALL_DIR = os.getenv("PYINTERAPH")

if not INSTALL_DIR:
    log.warn(\
        "PYINTERAPH system variable should contain the path to the " \
        "PYINTERAPH installation directory! Defaulting to local dir.")
    INSTALL_DIR = os.getcwd()

if not os.path.isdir(INSTALL_DIR):
    log.warn(\
        "The path specified by system variable PYINTERAPH does not " \
        "exist. Defaulting to local dir.")
    INSTALL_DIR = os.getcwd()

sys.path.append(INSTALL_DIR)


########################### ARGUMENT PARSER ###########################

description = "Interaction calculator"
parser = argparse.ArgumentParser(description = description)

#------------------------------ Top/traj -----------------------------#

s_helpstr = "Topology file"
parser.add_argument("-s", "--top", \
                    action = "store", \
                    type = str, \
                    dest = "top", \
                    default = None, \
                    help = s_helpstr)

t_helpstr = "Trajectory file"
parser.add_argument("-t", "--trj", \
                    action = "store", \
                    type = str, \
                    dest = "trj", \
                    default = None, \
                    help = t_helpstr)

r_helpstr = "Reference structure"
parser.add_argument("-r", "--ref", \
                    action = "store", \
                    type = str, \
                    dest = "ref", \
                    default = None, \
                    help = r_helpstr)

#------------------------------ Analyses -----------------------------#

b_helpstr = "Analyze salt-bridges"
parser.add_argument("-b", "--salt-bridges", \
                    action = "store_true", \
                    dest = "do_sb", \
                    help = b_helpstr)

f_helpstr = "Analyze hydrophobic clusters"
parser.add_argument("-f","--hydrophobic", \
                    action = "store_true", \
                    dest = "do_hc", \
                    help = f_helpstr)

y_helpstr = "Analyze hydrogen bonds"
parser.add_argument("-y","--hydrogen-bonds", \
                    action = "store_true", \
                    dest = "do_hb", \
                    help = y_helpstr)

p_helpstr = \
    "Analyze interactions using the knowledge-based potential"
parser.add_argument("-p","--potential", \
                    action = "store_true", \
                    dest = "do_kbp", \
                    help = p_helpstr)

#------------------------ Hydrophobic contacts -----------------------#

hc_reslist = ["ALA", "VAL", "LEU", "ILE", "PHE", "PRO", "TRP", "MET"]
hcres_helpstr = \
    "Comma-separated list of hydrophobic residues (default: {:s})"
parser.add_argument("--hc-residues", \
                    action = "store", \
                    nargs = "+", \
                    type = str, \
                    dest = "hc_reslist", \
                    default = hc_reslist, \
                    help = hcres_helpstr.format(\
                            ", ".join(hc_reslist)))

hcco_default = 5.0
hcco_helpstr = \
    "Distance cut-off for side-chain interaction (default: {:f})"
parser.add_argument("--hc-co", "--hc-cutoff", \
                    action = "store", \
                    dest = "hc_co", \
                    type = float, \
                    default = hcco_default, \
                    help = hcco_helpstr.format(hcco_default))

hcperco_default = 0.0
hcperco_helpstr = "Minimum persistence for interactions (default: {:f})"
parser.add_argument("--hc-perco", "--hc-persistence-cutoff", \
                    action = "store", \
                    type = float, \
                    dest = "hc_perco", \
                    default = 0.0, \
                    help = hcperco_helpstr.format(hcperco_default))

hcdat_default = "hydrophobic-clusters.dat"
hcdat_helpstr = \
    "Name of the file where to store the list of " \
    "hydrophobic contacts found (default: {:s})"
parser.add_argument("--hc-dat", \
                    action = "store", \
                    type = str, \
                    dest = "hc_dat", \
                    default = hcdat_default, \
                    help = hcdat_helpstr.format(hcdat_default))

hcgraph_helpstr = \
    "Name of the file where to store adjacency matrix " \
    "for interaction graph (hydrophobic contacts)"
parser.add_argument("--hc-graph", \
                    action = "store", \
                    dest = "hc_graph", \
                    type = str, \
                    default = None, \
                    help = hcgraph_helpstr)

#---------------------------- Salt bridges ---------------------------#

sbco_default = 4.5
sbco_helpstr = \
    "Distance cut-off for side-chain interaction (default: {:f})"
parser.add_argument("--sb-co", "--sb-cutoff", \
                    action = "store", \
                    type = float, \
                    dest = "sb_co", \
                    default = sbco_default, \
                    help = sbco_helpstr.format(sbco_default))

sbperco_default = 0.0
sbperco_helpstr = \
    "Minimum persistence for interactions (default: {:f})"
parser.add_argument("--sb-perco", "--sb-persistence-cutoff", \
                    action = "store", \
                    type = float, \
                    dest = "sb_perco", \
                    default = sbperco_default, \
                    help = sbperco_helpstr.format(sbperco_default))

sbdat_default = "salt-bridges.dat"
sbdat_helpstr = \
    "Name of the file where to store the list of " \
    "salt bridges found (default: {:s})"
parser.add_argument("--sb-dat", \
                    action = "store", \
                    type = str, \
                    dest = "sb_dat", \
                    default = sbdat_default, \
                    help = sbdat_helpstr.format(sbdat_default))

sbgraph_helpstr = \
    "Name of the file where to store adjacency matrix " \
    "for interaction graph (salt bridges)"
parser.add_argument("--sb-graph", \
                    action = "store", \
                    type = str, \
                    dest = "sb_graph", \
                    default = None, \
                    help = sbgraph_helpstr)

sbcgfile_default = os.path.join(INSTALL_DIR, "charged_groups.ini")
sbcgfile_helpstr = "Default charged groups file (default: {:s})"
parser.add_argument("--sb-cg-file", \
                    action = "store", \
                    type = str, \
                    dest = "cgs_file", \
                    default = sbcgfile_default, \
                    help = sbcgfile_helpstr.format(sbcgfile_default))

sbmode_choices = ["different_charge", "same_charge", "all"]
sbmode_default = "different_charge"
sbmode_helpstr = \
    "Electrostatic interactions mode. Accepted modes are {:s} " \
    "(default: {:s})"
parser.add_argument("--sb-mode", \
                    action = "store", \
                    type = str, \
                    dest = "sb_mode", \
                    choices = sbmode_choices, \
                    default = sbmode_default, \
                    help = sbmode_helpstr.format(\
                                ", ".join(sbmode_choices), \
                                sbmode_default))

#--------------------------- Hydrogen bonds --------------------------#

hbco_default = 3.5
hbco_helpstr = "Donor-acceptor distance cut-off (default: {:f})"
parser.add_argument("--hb-co", "--hb-cutoff", \
                    action = "store", \
                    type = float, \
                    dest = "hb_co", \
                    default = hbco_default, \
                    help = hbco_helpstr.format(hbco_default))

hbang_default = 120.0
hbang_helpstr = "Donor-acceptor angle cut-off (default: {:f})"
parser.add_argument("--hb-ang", "--hb-angle", \
                    action = "store", \
                    type = float, \
                    dest = "hb_angle", \
                    default = hbang_default, \
                    help = hbang_helpstr.format(hbang_default))

hbdat_default = "hydrogen-bonds.dat"
hbdat_helpstr = \
    "Name of the file where to store the list of " \
    "hydrogen bonds found (default: {:s})"
parser.add_argument("--hb-dat", \
                    action = "store", \
                    type = str, \
                    dest = "hb_dat", \
                    default = hbdat_default, \
                    help = hbdat_helpstr.format(hbdat_default))

hbgraph_helpstr = \
    "Name of the file where to store adjacency matrix " \
    "for interaction graph (hydrogen bonds)"
parser.add_argument("--hb-graph", \
                    action = "store", \
                    type = str, \
                    dest = "hb_graph", \
                    default = None, \
                    help = hbgraph_helpstr)

hbperco_default = 0.0
hbperco_helpstr = \
    "Minimum persistence for hydrogen bonds (default: {:f})"
parser.add_argument("--hb-perco", \
                    action = "store", \
                    type = float, \
                    dest = "hb_perco", \
                    default = hbperco_default, \
                    help = hbperco_helpstr.format(hbperco_default))

hbclass_choices = ["all", "mc-mc", "mc-sc", "sc-sc", "custom"]
hbclass_default = "all"
hbclass_helpstr = \
    "Class of hydrogen bonds to analyze. Accepted classes are {:s} " \
    "(default: {:s})"
parser.add_argument("--hb-class", \
                    action = "store", \
                    type = str, \
                    dest = "hb_class", \
                    choices = hbclass_choices, \
                    default = hbclass_default, \
                    help = hbclass_helpstr.format(\
                            ", ".join(hbclass_choices), \
                            hbclass_default))

hbadfile_default = os.path.join(INSTALL_DIR, "hydrogen_bonds.ini")
hbadfile_helpstr = \
    "File defining hydrogen bonds donor and acceptor atoms " \
    "(default: {:s})"
parser.add_argument("--hb-ad-file", \
                    action = "store", \
                    type = str, \
                    dest = "hbs_file", \
                    default = hbadfile_default, \
                    help = hbadfile_helpstr.format(hbadfile_default))

hbcustom1_helpstr = "Custom group 1 for hydrogen bonds calculation"
parser.add_argument("--hb-custom-group-1", \
                    action = "store", \
                    type = str, \
                    dest = "hb_group1", \
                    default = None, \
                    help = hbcustom1_helpstr)

hbcustom2_helpstr = "Custom group 2 for hydrogen bonds calculation"
parser.add_argument("--hb-custom-group-2", \
                    action = "store", \
                    type = str, \
                    dest = "hb_group2", \
                    default = None, \
                    help = hbcustom2_helpstr)

#----------------------------- Potential -----------------------------#

kbpff_default = os.path.join(INSTALL_DIR, "ff.S050.bin64")
kbpff_helpstr = "Statistical potential definition file (default: {:s})"
parser.add_argument("--kbp-ff", "--force-field", \
                    action = "store", \
                    type = str, \
                    dest = "kbp_ff", \
                    default = kbpff_default, \
                    help = kbpff_helpstr.format(kbpff_default))

kbpatom_default = os.path.join(INSTALL_DIR, "kbp_atomlist")
kbpatom_helpstr = \
    "Ordered, force-field specific list of atom names (default: {:s})"
parser.add_argument("--kbp-atomlist", \
                    action = "store", \
                    type = str, \
                    dest = "kbp_atomlist", \
                    default = kbpatom_default, \
                    help = kbpatom_helpstr.format(kbpatom_default))

kbpdat_default = "kb-potential.dat"
kbpdat_helpstr = \
    "Name of the file where to store the results of " \
    "the statistical potential analysis (default: {:s})"
parser.add_argument("--kbp-dat", \
                    action = "store", \
                    type = str, \
                    dest = "kbp_dat", \
                    default = kbpdat_default, \
                    help = kbpdat_helpstr.format(kbpdat_default))

kbpgraph_helpstr = \
    "Name of the file where to store adjacency matrix " \
    "for interaction graph (statistical potential)"
parser.add_argument('--kbp-graph', \
                    action = "store", \
                    type = str, \
                    dest = "kbp_graph", \
                    default = None, \
                    help = kbpgraph_helpstr)

kbpkbt_default = 1.0
kbpkbt_helpstr = \
    "kb*T value used in the inverse-Boltzmann relation " \
    "for the knowledge-based potential (default: {:f})"
parser.add_argument("--kbp-kbt", \
                    action = "store", \
                    type = float, \
                    dest = "kbp_kbt", \
                    default = kbpkbt_default, \
                    help = kbpkbt_helpstr.format(kbpkbt_default))

#---------------------------- Miscellanea ----------------------------#

ff_masses_dir = "ff_masses"
masses_dir = INSTALL_DIR + ff_masses_dir
masses_files = os.listdir(masses_dir)

ffmasses_default = "charmm27"
ffmasses_helpstr = \
    "Force field to be used (for masses calculation only). " \
    "Accepted force fields are {:s} (default: {:s})"
parser.add_argument("--ff-masses", \
                    action = "store", \
                    type = str, \
                    choices = masses_files, 
                    dest = "ffmasses", \
                    default = ffmasses_default, \
                    help = ffmasses_helpstr.format(\
                            ", ".join(masses_files), \
                            ffmasses_default))

v_helpstr = "Verbose mode"
parser.add_argument("-v", "--verbose", \
                    action = "store_true", \
                    dest = "verbose", \
                    help = v_helpstr)

args = parser.parse_args()


############################ LOGGER SETUP #############################

# Logging format
LOGFMT = "%(levelname)s: %(message)s"
# Verbose mode?
if args.verbose:
    log.basicConfig(level = log.INFO, format = LOGFMT)
else:
    log.basicConfig(level = log.WARNING, format = LOGFMT)


############################## ARGUMENTS ##############################

# input files
top = args.top
trj = args.trj
ref = args.ref
# hydrophobic contacts
do_hc = args.do_hc
hc_reslist = args.hc_reslist
hc_graph = args.hc_graph
hc_co = args.hc_co
hc_perco = args.hc_perco
hc_dat = args.hc_dat
# salt bridges
do_sb = args.do_sb
cgs_file = args.cgs_file
sb_mode = args.sb_mode
sb_graph = args.sb_graph
sb_co = args.sb_co
sb_perco = args.sb_perco
sb_dat = args.sb_dat
# hydrogen bonds
do_hb = args.do_hb
hbs_file = args.hbs_file
hb_group1 = args.hb_group1
hb_group2 = args.hb_group2
hb_class = args.hb_class
hb_graph = args.hb_graph
hb_co = args.hb_co
hb_perco = args.hb_perco
hb_angle = args.hb_angle
hb_dat = args.hb_dat
# potential
do_kbp = args.do_kbp
kbp_atomlist = args.kbp_atomlist
kbp_graph = args.kbp_graph
kbp_ff = args.kbp_ff
kbp_kbt = args.kbp_kbt
kbp_dat = args.kbp_dat
# miscellanea
ffmasses = os.path.join(masses_dir, args.ffmasses)


############################ CHECK INPUTS #############################

# top and trj must be present
if top is None or trj is None:
    log.error("Topology and trajectory are required.")
    exit(1)
# if no reference, topology is reference
if ref is None:
    ref = top
    log.info("Using topology as reference structure.")
# Load systems
try:
    pdb = mda.Universe(ref)
    uni = mda.Universe(top, trj)
except ValueError:
    logstr = \
        "Could not read one of the input files, or trajectory " \
        "and topology are not compatible."
    log.error(logstr)
    exit(1)


######################## HYDROPHOBIC CONTACTS #########################

if do_hc:
    fmfunc = None if hc_graph is None else li.calc_sc_fullmatrix
    str_out, hc_mat_out = li.do_interact(li.generate_sc_identifiers,
                                         pdb = pdb,
                                         uni = uni,
                                         co = hc_co, 
                                         perco = hc_perco,
                                         ffmasses = ffmasses, 
                                         calc_fullmatrix_func = fmfunc,
                                         residues_list = hc_reslist, \
                                         salt_bridges = False)

    # Save .dat
    with open(hc_dat, "w") as out:
        out.write(str_out)
    # Save .mat (if available)
    if hc_mat_out is not None:
        np.savetxt(hc_graph, hc_mat_out, fmt = "%.1f")


############################ SALT BRIDGES #############################

if do_sb:
    try:
        cgs = li.parse_cgs_file(cgs_file)
    except IOError:
        logstr = "Problems reading file {:s}."
        log.error(logstr.format(cgs_file), exc_info = True)
        exit(1)
    except:
        logstr = \
            "Could not parse the charged groups file {:s}. " \
            "Are there any inconsistencies?"
        log.error(logstr.format(cgs_file), exc_info = True)
        exit(1)
    
    if sb_mode == "same_charge":
        sb_mode = "same"
    elif sb_mode == "different_charge":
        sb_mode = "diff"
    elif sb_mode == "all":
        sb_mode = "both"

    fmfunc = None if sb_graph is None else li.calc_cg_fullmatrix
    str_out, sb_mat_out = li.do_interact(li.generate_cg_identifiers,
                                         pdb = pdb,
                                         uni = uni,
                                         co = sb_co, 
                                         perco = sb_perco,
                                         ffmasses = ffmasses, 
                                         fullmatrixfunc = fmfunc, 
                                         salt_bridges = True,
                                         salt_bridges_mode = sb_mode,
                                         cgs = cgs)

    # Save .dat
    with open(sb_dat, "w") as out:
        out.write(str_out)
    # Save .mat (if available)
    if sb_mat_out is not None:
        np.savetxt(sb_graph, sb_mat_out, fmt = "%.1f")


########################### HYDROGEN BONDS ############################

if args.do_hb:
    # atom selection for main chain hydrogen bonds
    mc_sel = "backbone or name H or name H1 or name H2 or name H3 " \
             "or name O1 or name O2 or name OXT"
    # atom selection for side chain hydrogen bonds
    sc_sel = "protein and not ({:s})".format(mc_sel)

    if (hb_group1 is not None and hb_group2 is not None) \
    and hb_class != "custom":
        warnstr = \
            "Hydrogen bond custom groups have been specified; " \
            "they will be used. Please use --hb-class=custom to " \
            "get rid of this warning!"
        log.warning(warnstr)    
        hb_class = "custom"

    if hb_class == "custom":
        if hb_group1 is None or hb_group2 is None:
            errstr = \
                "Hydrogen bond class 'custom' requires the " \
                "definition of two interation groups. (see " \
                "options --hb-custom-group1 and --hb-custom-group2)"
            log.error(errstr)
            exit(1)

    elif hb_class == "all":
        hb_group1 = "protein"
        hb_group2 = "protein"
    
    elif hb_class == "mc-mc":
        hb_group1 = mc_sel
        hb_group2 = mc_sel
    
    elif hb_class == "sc-sc":
        hb_group1 = sc_sel
        hb_group2 = sc_sel
    
    elif hb_class == "mc-sc":
        hb_group1 = mc_sel
        hb_group2 = sc_sel

    # check if selection 1 is valid
    try:
        uni.select_atoms(hb_group1)
    except:
        log.error("Selection 1 is invalid", exc_info = True)
        exit(1)
    # check if selection 2 is valid
    try:
        uni.select_atoms(hb_group2)
    except:
        log.error("Selection 2 is invalid", exc_info = True)
        exit(1)
    # check the donors-acceptors file
    try:
        hbs = li.parse_hbs_file(hbs_file)
    except IOError:
        logstr = "Problems reading file {:s}."
        log.error(logstr.format(hbs_file), exc_info = True)
        exit(1)
    except:
        logstr = \
            "Could not parse the hydrogen bonds file {:s}. " \
            "Are there any inconsistencies?"
        log.error(logstr.format(hbs_file), exc_info = True)
        exit(1)

    dofullmatrix = False if hb_graph is None else True
    perresidue = False    
    str_out, hb_mat_out = li.do_hbonds(sel1 = hb_group1,
                                       sel2 = hb_group2,
                                       pdb = pdb,
                                       uni = uni,
                                       distance = hb_co,
                                       angle = hb_angle,
                                       perco = hb_perco,
                                       dofullmatrix = dofullmatrix,
                                       other_hbs = hbs, \
                                       perresidue = False)                                    

    # Save .dat
    with open(hb_dat, "w") as out:
        out.write(str_out)
    # Save .mat (if available)
    if hb_mat_out is not None:
        np.savetxt(hb_graph, hb_mat_out, fmt = "%.1f")


######################## STATISTICAL POTENTIAL ########################

if do_kbp:
    # Residue list for potential calculation - all canonical but GLY
    kbp_residues_list = \
        ["ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "HIS", \
         "ILE", "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", \
         "TRP", "TYR", "VAL"]
    
    kbp_atomlist = li.parse_atomlist(kbp_atomlist)
    dofullmatrix = False if kbp_graph is None else True
    str_out, kbp_mat_out = li.do_potential(kbp_atomlist = kbp_atomlist, 
                                           residues_list = kbp_residues_list,
                                           potential_file = kbp_ff,
                                           uni = uni,
                                           pdb = pdb,
                                           seq_dist_co = 0, 
                                           dofullmatrix = dofullmatrix,
                                           kbT = kbp_kbt)

    # Save .dat
    with open(kbp_dat, "w") as out:
        out.write(str_out)
    # Save .mat (if available)
    if mat_out is not None:
        np.savetxt(kbp_graph, kbp_mat_out, fmt = "%.3f")
