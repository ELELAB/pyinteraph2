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
import pkg_resources
import MDAnalysis as mda
import numpy as np
from libinteract import libinteract as li

def main():


    ######################### ARGUMENT PARSER #########################


    description = "Interaction calculator"
    parser = argparse.ArgumentParser(description = description)


    #---------------------------- Top/traj ---------------------------#


    s_helpstr = "Topology file"
    parser.add_argument("-s", "--top",
                        action = "store",
                        type = str,
                        dest = "top",
                        default = None,
                        help = s_helpstr)

    t_helpstr = "Trajectory file"
    parser.add_argument("-t", "--trj",
                        action = "store",
                        type = str,
                        dest = "trj",
                        default = None,
                        help = t_helpstr)

    r_helpstr = "Reference structure"
    parser.add_argument("-r", "--ref",
                        action = "store",
                        type = str,
                        dest = "ref",
                        default = None,
                        help = r_helpstr)


    #---------------------------- Analyses ---------------------------#


    m_helpstr = "Analyze side-chain centers-of-mass contacts (cmPSN)"
    parser.add_argument("-m", "--cmpsn",
                        action = "store_true",
                        dest = "do_cmpsn",
                        help = m_helpstr)

    a_helpstr = "Analyze atomic contacts (acPSN)"
    parser.add_argument("-a", "--acpsn",
                        action = "store_true",
                        dest = "do_acpsn",
                        help = a_helpstr)

    f_helpstr = "Analyze hydrophobic clusters"
    parser.add_argument("-f", "--hydrophobic",
                        action = "store_true",
                        dest = "do_hc",
                        help = f_helpstr)

    b_helpstr = "Analyze salt bridges"
    parser.add_argument("-b", "--salt-bridges",
                        action = "store_true",
                        dest = "do_sb",
                        help = b_helpstr)

    y_helpstr = "Analyze hydrogen bonds"
    parser.add_argument("-y","--hydrogen-bonds",
                        action = "store_true",
                        dest = "do_hb",
                        help = y_helpstr)

    p_helpstr = \
        "Analyze interactions using the knowledge-based potential"
    parser.add_argument("-p","--potential",
                        action = "store_true",
                        dest = "do_kbp",
                        help = p_helpstr)


    #----------------------------- cmPSN -----------------------------#


    cmpsnco_default = 5.0
    cmpsnco_helpstr = \
        f"Distance cut-off for the cmPSN (default: {cmpsnco_default})"
    parser.add_argument("--cmpsn-co", "--cmpsn-cutoff",
                        action = "store",
                        dest = "cmpsn_co",
                        type = float,
                        default = cmpsnco_default,
                        help = cmpsnco_helpstr)

    cmpsnperco_default = 0.0
    cmpsnperco_helpstr = \
        f"Minimum persistence for the cmPSN (default: " \
        f"{cmpsnperco_default})"
    parser.add_argument("--cmpsn-perco", "--cmpsn-persistence-cutoff",
                        action = "store",
                        type = float,
                        dest = "cmpsn_perco",
                        default = cmpsnperco_default,
                        help = cmpsnperco_helpstr)

    cmpsn_reslist = \
        ["ALA", "CYS", "ASP", "GLU", "PHE", "HIS", "ILE", "LYS", "LEU",
         "MET", "ASN", "PRO", "GLN", "ARG", "SER", "THR", "VAL", "TRP",
         "TYR"]
    cmpsnres_helpstr = \
        f"Comma-separated list of residues to be used when " \
        f"calculating the cmPSN (default: {', '.join(cmpsn_reslist)})"
    parser.add_argument("--cmpsn-residues",
                        action = "store",
                        type = str,
                        dest = "cmpsn_reslist",
                        default = cmpsn_reslist,
                        help = cmpsnres_helpstr)

    cmpsn_correction_choices = ["null", "rg"]
    cmpsn_correction_default = "null"
    cmpsn_correction_helpstr = \
        f"Correction to be applied to the cmPSN (default: " \
        f"{cmpsn_correction_default}"
    parser.add_argument("--cmpsn-correction",
                        action = "store",
                        dest = "cmpsn_correction",
                        type = str,
                        choices = cmpsn_correction_choices, 
                        default = cmpsn_correction_default,
                        help = cmpsn_correction_helpstr)

    cmpsncsv_default = "cmpsn.csv"
    cmpsncsv_helpstr = \
        f"Name of the CSV file where to store the list of contacts " \
        f"found in the cmPSN (default: {cmpsncsv_default})"
    parser.add_argument("--cmpsn-csv",
                        action = "store",
                        type = str,
                        dest = "cmpsn_csv",
                        default = cmpsncsv_default,
                        help = cmpsncsv_helpstr)

    cmpsngraph_helpstr = \
        "File where to store the adjacency matrix for the " \
        "interaction graph (cmPSN)"
    parser.add_argument("--cmpsn-graph",
                        action = "store",
                        dest = "cmpsn_graph",
                        type = str,
                        default = None,
                        help = cmpsngraph_helpstr)


    #----------------------------- acPSN -----------------------------#


    acpsnco_default = 4.5
    acpsnco_helpstr = \
        f"Distance cut-off for the acPSN (default: {acpsnco_default})"
    parser.add_argument("--acpsn-co", "--acpsn-cutoff",
                        action = "store",
                        dest = "acpsn_co",
                        type = float,
                        default = acpsnco_default,
                        help = acpsnco_helpstr)

    acpsnperco_default = 0.0
    acpsnperco_helpstr = \
        f"Minimum persistence for the acPSN (default: " \
        f"{acpsnperco_default})"
    parser.add_argument("--acpsn-perco", "--acpsn-persistence-cutoff",
                        action = "store",
                        type = float,
                        dest = "acpsn_perco",
                        default = acpsnperco_default,
                        help = acpsnperco_helpstr)

    acpsnproxco_default = 1
    acpsnproxco_helpstr = \
        f"Minimum sequence distance for the acPSN (default: " \
        f"{acpsnproxco_default})"
    parser.add_argument("--acpsn-proxco", "--acpsn-sequence-cutoff",
                        action = "store",
                        type = int,
                        dest = "acpsn_proxco",
                        default = acpsnproxco_default,
                        help = acpsnproxco_helpstr)

    acpsnimin_default = 3.0
    acpsnimin_helpstr = \
        f"Minimum interaction strength value for the acPSN " \
        f"(default: {acpsnimin_default})"
    parser.add_argument("--acpsn-imin", "--acpsn-imin-cutoff",
                        action = "store",
                        type = float,
                        dest = "acpsn_imin",
                        default = acpsnimin_default,
                        help = acpsnimin_helpstr)

    acpsnew_choices = ["strength", "persistence"]
    acpsnew_default = "strength"
    acpsnew_helpstr = \
        f"Edge weighting method for the acPSN (default: " \
        f"{acpsnew_default})"
    parser.add_argument("--acpsn-ew", "--acpsn-edge-weights",
                        action = "store",
                        type = str,
                        dest = "acpsn_ew",
                        choices = acpsnew_choices,
                        default = acpsnew_default,
                        help = acpsnew_helpstr)

    acpsnnffile_default = \
        pkg_resources.resource_filename("pyinteraph",
                                        "normalization_factors.ini")
    acpsnnffile_helpstr = \
        f"File with normalization factors to be used " \
        f"in the calculation of the acPSN (default: " \
        f"{acpsnnffile_default})"
    parser.add_argument("--acpsn-nf-file",
                        action = "store",
                        type = str,
                        dest = "nf_file",
                        default = acpsnnffile_default,
                        help = acpsnnffile_helpstr)

    acpsnnf_default = 999.9
    acpsnnfpermissive_helpstr = \
        f"Permissive mode. If a residue with no associated " \
        f"normalization factor is found, the default normalization " \
        f"factor {acpsnnf_default} will be used, and no error will " \
        f"be thrown"
    parser.add_argument("--acpsn-nf-permissive",
                        action = "store_true",
                        dest = "nf_permissive",
                        help = acpsnnfpermissive_helpstr)

    acpsnnf_helpstr = \
        f"Default normalization factor to be used when running in " \
        f"permissive mode (default: {acpsnnf_default})"
    parser.add_argument("--acpsn-nf-default",
                        action = "store",
                        type = str,
                        dest = "nf_default",
                        default = acpsnnf_default,
                        help = acpsnnf_helpstr)

    acpsncsv_default = "acpsn.csv"
    acpsncsv_helpstr = \
        f"Name of the CSV file where to store the list of contacts " \
        f"found in the acPSN (default: {acpsncsv_default})"
    parser.add_argument("--acpsn-csv",
                        action = "store",
                        type = str,
                        dest = "acpsn_csv",
                        default = acpsncsv_default,
                        help = acpsncsv_helpstr)

    acpsngraph_helpstr = \
        "File where to store the adjacency matrix for the " \
        "interaction graph (acPSN)"
    parser.add_argument("--acpsn-graph",
                        action = "store",
                        dest = "acpsn_graph",
                        type = str,
                        default = None,
                        help = acpsngraph_helpstr)


    #---------------------- Hydrophobic contacts ---------------------#


    hcco_default = 5.0
    hcco_helpstr = \
        f"Distance cut-off for hydrophobic contacts (default: " \
        f"{hcco_default})"
    parser.add_argument("--hc-co", "--hc-cutoff",
                        action = "store",
                        dest = "hc_co",
                        type = float,
                        default = hcco_default,
                        help = hcco_helpstr)

    hcperco_default = 0.0
    hcperco_helpstr = \
        f"Minimum persistence for hydrophobic contacts (default: " \
        f"{hcperco_default})"
    parser.add_argument("--hc-perco", "--hc-persistence-cutoff",
                        action = "store",
                        type = float,
                        dest = "hc_perco",
                        default = hcperco_default,
                        help = hcperco_helpstr)

    hc_reslist = \
        ["ALA", "VAL", "LEU", "ILE", "PHE", "PRO", "TRP", "MET"]
    hcres_helpstr = \
        f"Comma-separated list of hydrophobic residues (default: " \
        f"{', '.join(hc_reslist)})"
    parser.add_argument("--hc-residues",
                        action = "store",
                        type = str,
                        dest = "hc_reslist",
                        default = hc_reslist,
                        help = hcres_helpstr)

    hccsv_default = "hydrophobic-clusters.csv"
    hccsv_helpstr = \
        f"Name of the CSV file where to store the list of " \
        f"hydrophobic contacts found (default: {hccsv_default})"
    parser.add_argument("--hc-csv",
                        action = "store",
                        type = str,
                        dest = "hc_csv",
                        default = hccsv_default,
                        help = hccsv_helpstr)

    hcgraph_helpstr = \
        "File where to store the adjacency matrix for the " \
        "interaction graph (hydrophobic contacts)"
    parser.add_argument("--hc-graph",
                        action = "store",
                        dest = "hc_graph",
                        type = str,
                        default = None,
                        help = hcgraph_helpstr)


    #-------------------------- Salt bridges -------------------------#


    sbco_default = 4.5
    sbco_helpstr = \
        f"Distance cut-off for salt bridges (default: {sbco_default})"
    parser.add_argument("--sb-co", "--sb-cutoff",
                        action = "store",
                        type = float,
                        dest = "sb_co",
                        default = sbco_default,
                        help = sbco_helpstr)

    sbperco_default = 0.0
    sbperco_helpstr = \
        f"Minimum persistence for salt bridges (default: " \
        f"{sbperco_default})"
    parser.add_argument("--sb-perco", "--sb-persistence-cutoff",
                        action = "store",
                        type = float,
                        dest = "sb_perco",
                        default = sbperco_default,
                        help = sbperco_helpstr)

    sbmode_choices = ["different_charge", "same_charge", "all"]
    sbmode_default = "different_charge"
    sbmode_helpstr = \
        f"Electrostatic interactions mode (default: {sbmode_default})"
    parser.add_argument("--sb-mode",
                        action = "store",
                        type = str,
                        dest = "sb_mode",
                        choices = sbmode_choices,
                        default = sbmode_default,
                        help = sbmode_helpstr)

    sbcgfile_default = \
        pkg_resources.resource_filename("pyinteraph",
                                        "charged_groups.ini")
    sbcgfile_helpstr = \
        f"File with charged groups to be used to find " \
        f"salt bridges (default: {sbcgfile_default})"
    parser.add_argument("--sb-cg-file",
                        action = "store",
                        type = str,
                        dest = "cgs_file",
                        default = sbcgfile_default,
                        help = sbcgfile_helpstr)

    sbcsv_default = "salt-bridges.csv"
    sbcsv_helpstr = \
        f"Name of the CSV file where to store the list of " \
        f"salt bridges found (default: {sbcsv_default})"
    parser.add_argument("--sb-csv",
                        action = "store",
                        type = str,
                        dest = "sb_csv",
                        default = sbcsv_default,
                        help = sbcsv_helpstr)

    sbgraph_helpstr = \
        "File where to store the adjacency matrix for the " \
        "interaction graph (salt bridges)"
    parser.add_argument("--sb-graph",
                        action = "store",
                        type = str,
                        dest = "sb_graph",
                        default = None,
                        help = sbgraph_helpstr)


    #------------------------- Hydrogen bonds ------------------------#


    # Default taken from the default used by MDAnalysis (v2.0.0)
    hbco_default = 3.0
    hbco_helpstr = \
        f"Donor-acceptor distance cut-off for hydrogen bonds " \
        f"(default: {hbco_default})"
    parser.add_argument("--hb-co", "--hb-cutoff",
                        action = "store",
                        type = float,
                        dest = "hb_co",
                        default = hbco_default,
                        help = hbco_helpstr)
    
    # Default taken from the default used by MDAnalysis (v2.0.0)
    hbang_default = 150.0
    hbang_helpstr = \
        f"Donor-acceptor angle cut-off for hydrogen bonds " \
        f"(default: {hbang_default})"
    parser.add_argument("--hb-ang", "--hb-angle",
                        action = "store",
                        type = float,
                        dest = "hb_angle",
                        default = hbang_default,
                        help = hbang_helpstr)

    hbperco_default = 0.0
    hbperco_helpstr = \
        f"Minimum persistence for hydrogen bonds (default: " \
        f"{hbperco_default})"
    parser.add_argument("--hb-perco",
                        action = "store",
                        type = float,
                        dest = "hb_perco",
                        default = hbperco_default,
                        help = hbperco_helpstr)

    hbclass_choices = ["all", "mc-mc", "mc-sc", "sc-sc", "custom"]
    hbclass_default = "all"
    hbclass_helpstr = \
        f"Class of hydrogen bonds to analyze (default: " \
        f"{hbclass_default})"
    parser.add_argument("--hb-class",
                        action = "store",
                        type = str,
                        dest = "hb_class",
                        choices = hbclass_choices,
                        default = hbclass_default,
                        help = hbclass_helpstr)

    hbcustom1_helpstr = "Custom group 1 for hydrogen bonds calculation"
    parser.add_argument("--hb-custom-group-1",
                        action = "store",
                        type = str,
                        dest = "hb_group1",
                        default = None,
                        help = hbcustom1_helpstr)

    hbcustom2_helpstr = "Custom group 2 for hydrogen bonds calculation"
    parser.add_argument("--hb-custom-group-2",
                        action = "store",
                        type = str,
                        dest = "hb_group2",
                        default = None,
                        help = hbcustom2_helpstr)

    hbadfile_default = \
        pkg_resources.resource_filename("pyinteraph",
                                        "hydrogen_bonds.ini")
    hbadfile_helpstr = \
        f"File defining hydrogen bonds donor and acceptor atoms " \
        f"(default: {hbadfile_default})"
    parser.add_argument("--hb-ad-file",
                        action = "store",
                        type = str,
                        dest = "hbs_file",
                        default = hbadfile_default,
                        help = hbadfile_helpstr)

    hbcsv_default = "hydrogen-bonds.csv"
    hbcsv_helpstr = \
        f"Name of the CSV file where to store the list of hydrogen " \
        f"bonds found (default: {hbcsv_default})"
    parser.add_argument("--hb-csv",
                        action = "store",
                        type = str,
                        dest = "hb_csv",
                        default = hbcsv_default,
                        help = hbcsv_helpstr)

    hbgraph_helpstr = \
        "File where to store the adjacency matrix for the " \
        "interaction graph (hydrogen bonds)"
    parser.add_argument("--hb-graph",
                        action = "store",
                        type = str,
                        dest = "hb_graph",
                        default = None,
                        help = hbgraph_helpstr)


    #--------------------------- Potential ---------------------------#


    kbpkbt_default = 1.0
    kbpkbt_helpstr = \
        f"kb*T value used in the inverse-Boltzmann relation for the " \
        f"knowledge-based potential (default: {kbpkbt_default})"
    parser.add_argument("--kbp-kbt",
                        action = "store",
                        type = float,
                        dest = "kbp_kbt",
                        default = kbpkbt_default,
                        help = kbpkbt_helpstr)
    

    kbpff_default = \
        pkg_resources.resource_filename("pyinteraph", "ff.S050.bin64")
    kbpff_helpstr = \
        f"Statistical potential definition file (default: " \
        f"{hbperco_default})"
    parser.add_argument("--kbp-ff", "--force-field",
                        action = "store",
                        type = str,
                        dest = "kbp_ff",
                        default = kbpff_default,
                        help = kbpff_helpstr)

    kbpatom_default = \
        pkg_resources.resource_filename("pyinteraph", "kbp_atomlist")
    kbpatom_helpstr = \
        f"Ordered, force-field specific list of atom names " \
        f"(default: {kbpatom_default})"
    parser.add_argument("--kbp-atomlist",
                        action = "store",
                        type = str,
                        dest = "kbp_atomlist",
                        default = kbpatom_default,
                        help = kbpatom_helpstr)

    kbpcsv_default = "kb-potential.csv"
    kbpcsv_helpstr = \
        f"File where to store the results of the statistical " \
        f"potential analysis (default: {kbpcsv_default})"
    parser.add_argument("--kbp-csv",
                        action = "store",
                        type = str,
                        dest = "kbp_csv",
                        default = kbpcsv_default,
                        help = kbpcsv_helpstr)

    kbpgraph_helpstr = \
        "File where to store the adjacency matrix for the " \
        "interaction graph (statistical potential)"
    parser.add_argument("--kbp-graph",
                        action = "store",
                        type = str,
                        dest = "kbp_graph",
                        default = None,
                        help = kbpgraph_helpstr)


    #-------------------------- Miscellanea --------------------------#


    ffmasses_dir = "ff_masses"
    ffmasses_dir = \
        pkg_resources.resource_filename("pyinteraph", ffmasses_dir)
    
    ffmasses_choices = os.listdir(ffmasses_dir)
    ffmasses_default = "charmm27"
    ffmasses_helpstr = \
        f"Force field to be used (for masses calculation only) " \
        f"(default: {ffmasses_default})"
    parser.add_argument("--ff-masses",
                        action = "store",
                        type = str,
                        dest = "ffmasses",
                        choices = ffmasses_choices, 
                        default = ffmasses_default,
                        help = ffmasses_helpstr)

    v_helpstr = "Verbose mode"
    parser.add_argument("-v", "--verbose",
                        action = "store_true",
                        dest = "verbose",
                        help = v_helpstr)

    args = parser.parse_args()


    ########################## LOGGER SETUP ###########################


    # Logging format
    LOGFMT = "%(levelname)s: %(message)s"
    
    # Verbose mode?
    if args.verbose:
        log.basicConfig(level = log.INFO, format = LOGFMT)
    else:
        log.basicConfig(level = log.WARNING, format = LOGFMT)


    ############################ ARGUMENTS ############################


    # Input files
    top = args.top
    trj = args.trj
    ref = args.ref

    # cmPSN
    do_cmpsn = args.do_cmpsn
    cmpsn_co = args.cmpsn_co
    cmpsn_perco = args.cmpsn_perco
    if type(args.cmpsn_reslist) is str:
        cmpsn_reslist = [l.strip() for l in args.cmpsn_reslist.split(",")]
    else:
        cmpsn_reslist = args.cmpsn_reslist
    cmpsn_correction = args.cmpsn_correction
    cmpsn_csv = args.cmpsn_csv
    cmpsn_graph = args.cmpsn_graph

    # acPSN
    do_acpsn = args.do_acpsn
    acpsn_co = args.acpsn_co
    acpsn_perco = args.acpsn_perco
    acpsn_proxco = args.acpsn_proxco
    acpsn_imin = args.acpsn_imin
    acpsn_ew = args.acpsn_ew
    nf_file = args.nf_file
    nf_permissive = args.nf_permissive
    nf_default = args.nf_default
    acpsn_csv = args.acpsn_csv
    acpsn_graph = args.acpsn_graph
    
    # Hydrophobic contacts
    do_hc = args.do_hc
    hc_co = args.hc_co
    hc_perco = args.hc_perco
    if type(args.hc_reslist) is str:
        hc_reslist = [l.strip() for l in args.hc_reslist.split(",")]
    else:
        hc_reslist = args.hc_reslist
    hc_csv = args.hc_csv
    hc_graph = args.hc_graph
    
    # Salt bridges
    do_sb = args.do_sb
    sb_co = args.sb_co
    sb_perco = args.sb_perco
    sb_mode = args.sb_mode
    cgs_file = args.cgs_file
    sb_csv = args.sb_csv
    sb_graph = args.sb_graph
    
    # Hydrogen bonds
    do_hb = args.do_hb
    hb_angle = args.hb_angle
    hb_co = args.hb_co
    hb_perco = args.hb_perco
    hb_class = args.hb_class
    hb_group1 = args.hb_group1
    hb_group2 = args.hb_group2
    hbs_file = args.hbs_file
    hb_csv = args.hb_csv
    hb_graph = args.hb_graph
   
    # Statistical potential
    do_kbp = args.do_kbp
    kbp_kbt = args.kbp_kbt
    kbp_atomlist = args.kbp_atomlist
    kbp_ff = args.kbp_ff
    kbp_csv = args.kbp_csv
    kbp_graph = args.kbp_graph
    
    # Miscellanea
    ffmasses = os.path.join(ffmasses_dir, args.ffmasses)


    ########################## CHECK INPUTS ###########################


    # Topology and trajectory must be present
    if not top or not trj:
        log.error("Topology and trajectory are required.")
        exit(1)
    
    # If no reference structure is passed, the topology will be 
    # the reference
    if not ref:
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


    ###################### HYDROPHOBIC CONTACTS #######################


    if do_hc:

        # Function to compute the full matrix
        hc_fmfunc = \
            None if hc_graph is None else li.calc_sc_fullmatrix

        # Compute the table and the matrix
        hc_table_out, hc_mat_out = \
            li.do_interact(identfunc = li.generate_sc_identifiers,
                           pdb = pdb,
                           uni = uni,
                           co = hc_co,
                           perco = hc_perco,
                           ffmasses = ffmasses,
                           fullmatrixfunc = hc_fmfunc,
                           mindist = False,
                           reslist = hc_reslist,
                           correction_func = li.null_correction)

        # Save .csv
        hc_table_dict = li.create_dict_tables(hc_table_out)
        li.save_output_dict(hc_table_dict, hc_csv)

        # Save .dat (if available) (hc_graph being not None has
        # been checked already)
        if hc_mat_out is not None:
            hc_mat_dict = \
                li.create_dict_matrices(hc_mat_out,
                                        hc_table_dict,
                                        pdb)
            li.save_output_dict(hc_mat_dict, hc_graph)


    ############################## cmPSN ##############################


    if do_cmpsn:

        # Function to compute the full matrix
        cmpsn_fmfunc = \
            None if not cmpsn_graph else li.calc_sc_fullmatrix

        # Select the appropriate correction function
        if cmpsn_correction == "null":
            selected_func = li.null_correction
        elif cmpsn_correction == "rg":
            selected_func = li.rg_correction

        # Compute the table and the matrix
        cmpsn_table_out, cmpsn_mat_out = \
            li.do_interact(identfunc = li.generate_sc_identifiers,
                           pdb = pdb,
                           uni = uni,
                           co = cmpsn_co,
                           perco = cmpsn_perco,
                           ffmasses = ffmasses,
                           fullmatrixfunc = cmpsn_fmfunc,
                           mindist = False,
                           reslist = cmpsn_reslist,
                           correction_func = selected_func)

        # Save .csv
        cmpsn_table_dict = li.create_dict_tables(cmpsn_table_out)
        li.save_output_dict(cmpsn_table_dict, cmpsn_csv)

        # Save .dat (if available)
        if cmpsn_mat_out is not None and cmpsn_graph is not None:
            cmpsn_mat_dict = \
                li.create_dict_matrices(cmpsn_mat_out, 
                                        cmpsn_table_dict,
                                        pdb)
            li.save_output_dict(cmpsn_mat_dict, cmpsn_graph)


    ############################## acPSN ##############################


    if do_acpsn:

        # Try to parse the charged groups definitions file
        try:
            norm_facts = li.parse_nf_file(nf_file)
        except IOError:
            logstr = f"Problems reading file {nf_file}."
            log.error(logstr, exc_info = True)
            exit(1)
        except:
            logstr = \
                f"Could not parse the normalization factors file " \
                f"{nf_file}. Are there any inconsistencies?"
            log.error(logstr, exc_info = True)
            exit(1)

        # Try to compute the table of contacts and the matrix
        try:
            acpsn_table_out, acpsn_mat_out = \
                li.do_acpsn(pdb = pdb,
                            uni = uni,
                            co = acpsn_co,
                            perco = acpsn_perco,
                            proxco = acpsn_proxco,
                            imin = acpsn_imin,
                            edge_weight = acpsn_ew,
                            norm_facts = norm_facts,
                            nf_permissive = nf_permissive,
                            nf_default = nf_default)
        # In case something went wront, report it and exit
        except Exception as e:
            logstr = \
                f"Could not compute the table of contacts and the " \
                f"matrix for acPSN: {e}"
            log.error(logstr)
            exit(1)

        # Save .csv
        acpsn_table_dict = li.create_dict_tables(acpsn_table_out)
        li.save_output_dict(acpsn_table_dict, acpsn_csv)

        # Save .dat (if available)
        if acpsn_mat_out is not None and acpsn_graph is not None:
            acpsn_mat_dict = \
                li.create_dict_matrices(acpsn_mat_out, 
                                        acpsn_table_dict,
                                        pdb)
            li.save_output_dict(acpsn_mat_dict, acpsn_graph)


    ########################## SALT BRIDGES ###########################


    if do_sb:
        
        # Try to parse the charged groups definitions file
        try:
            cgs = li.parse_cgs_file(cgs_file)
        except IOError:
            logstr = f"Problems reading file {cgs_file}."
            log.error(logstr, exc_info = True)
            exit(1)
        except:
            logstr = \
                f"Could not parse the charged groups file " \
                f"{cgs_file}. Are there any inconsistencies?"
            log.error(logstr, exc_info = True)
            exit(1)
        
        if sb_mode == "same_charge":
            sb_mode = "same"
        elif sb_mode == "different_charge":
            sb_mode = "diff"
        elif sb_mode == "all":
            sb_mode = "both"

        # Function to compute the full matrix
        sb_fmfunc = None if not sb_graph else li.calc_cg_fullmatrix

        # Compute the table and the matrix
        sb_table_out, sb_mat_out = \
            li.do_interact(identfunc = li.generate_cg_identifiers,
                           pdb = pdb,
                           uni = uni,
                           co = sb_co, 
                           perco = sb_perco,
                           ffmasses = ffmasses, 
                           fullmatrixfunc = sb_fmfunc, 
                           mindist = True,
                           mindist_mode = sb_mode,
                           cgs = cgs)

        # Save .csv
        sb_table_dict = li.create_dict_tables(sb_table_out)
        li.save_output_dict(sb_table_dict, sb_csv)

        # Save .dat (if available)
        if sb_mat_out is not None and sb_graph is not None:
            sb_mat_dict = \
                li.create_dict_matrices(sb_mat_out,
                                        sb_table_dict,
                                        pdb)
            li.save_output_dict(sb_mat_dict, sb_graph)


    ########################### HYDROGEN BONDS ############################


    if do_hb:
        
        # Atom selection for main chain hydrogen bonds
        mc_sel = "backbone or name H or name H1 or name H2 " \
                 "or name H3 or name O1 or name O2 or name OXT"
        
        # Atom selection for side chain hydrogen bonds
        sc_sel = f"protein and (not {mc_sel})"

        # Custom atom selection
        if (hb_group1 and hb_group2) and hb_class != "custom":
            warnstr = \
                "Hydrogen bond custom groups have been specified; " \
                "they will be used. Please use --hb-class=custom to " \
                "get rid of this warning!"
            log.warning(warnstr)    
            hb_class = "custom"

        if hb_class == "custom":
            if not hb_group1 or not hb_group2:
                errstr = \
                    "Hydrogen bond class 'custom' requires the " \
                    "definition of two interation groups. (see " \
                    "options --hb-custom-group1 and --hb-custom-group2)"
                log.error(errstr)
                exit(1)

        # All hydrogen bonds
        elif hb_class == "all":
            hb_group1 = "protein"
            hb_group2 = "protein"
        
        # Main chain - main chain hydrogen bonds
        elif hb_class == "mc-mc":
            hb_group1 = mc_sel
            hb_group2 = mc_sel
        
        # Side chain - side chain hydrogen bonds
        elif hb_class == "sc-sc":
            hb_group1 = sc_sel
            hb_group2 = sc_sel
        
        # Main chain - side chain hydrogen bonds
        elif hb_class == "mc-sc":
            hb_group1 = mc_sel
            hb_group2 = sc_sel

        # Check if selection 1 is valid
        try:
            uni.select_atoms(hb_group1)
        except:
            log.error("Selection 1 is invalid", exc_info = True)
            exit(1)
        
        # Check if selection 2 is valid
        try:
            uni.select_atoms(hb_group2)
        except:
            log.error("Selection 2 is invalid", exc_info = True)
            exit(1)
        
        # Check the donors-acceptors file
        try:
            hbs = li.parse_hbs_file(hbs_file)
        except IOError:
            logstr = f"Problems reading {hbs_file}."
            log.error(logstr, exc_info = True)
            exit(1)
        except:
            logstr = f"Could not parse {hbs_file}." 
            log.error(logstr, exc_info = True)
            exit(1)

        # Whether to compute the full matrix
        do_fullmatrix = True if hb_graph else False

        # Compute the table and the matrix
        hb_table_out, hb_mat_out = \
            li.do_hbonds(sel1 = hb_group1,
                         sel2 = hb_group2,
                         pdb = pdb,
                         uni = uni,
                         update_selections = True,
                         d_a_cutoff = hb_co,
                         d_h_a_angle_cutoff = hb_angle,
                         perco = hb_perco,
                         do_fullmatrix = do_fullmatrix,
                         other_hbs = hbs)                                    

        # Save .csv
        hb_table_dict = li.create_dict_tables(hb_table_out)
        li.save_output_dict(hb_table_dict, hb_csv)

        # Save .dat (if available)
        if hb_mat_out is not None and hb_graph is not None:
            hb_mat_dict = \
                li.create_dict_matrices(hb_mat_out,
                                        hb_table_dict,
                                        pdb)
            li.save_output_dict(hb_mat_dict, hb_graph)


    ######################## STATISTICAL POTENTIAL ########################


    if do_kbp:
        
        # Residue list for potential calculation - all canonical but GLY
        kbp_reslist = \
            ["ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "HIS",
             "ILE", "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR",
             "TRP", "TYR", "VAL"]
        
        # Parse the atom list
        kbp_atomlist = li.parse_atomlist(kbp_atomlist)

        # Whether to compute the full matrix
        do_fullmatrix = True if kbp_graph else False

        # Compute the output table and the matrix
        kbp_table_out, kbp_mat_out = \
            li.do_potential(kbp_atomlist = kbp_atomlist,
                            residues_list = kbp_reslist,
                            potential_file = kbp_ff,
                            uni = uni,
                            pdb = pdb,
                            do_fullmatrix = do_fullmatrix,
                            kbT = kbp_kbt,
                            seq_dist_co = 0)

        # Save .csv
        kbp_table_dict = li.create_dict_tables(kbp_table_out)
        li.save_output_dict(kbp_table_dict, kbp_csv)
        
        # Save .mat (if available)
        if kbp_mat_out is not None and kbp_graph is not None:
            np.savetxt(kbp_graph, kbp_mat_out, fmt = "%.3f")


if __name__ == "__main__":
    main()
