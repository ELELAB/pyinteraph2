#!/usr/bin/env python
# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-

import MDAnalysis as mda
import os
import os.path
import numpy as np
import pytest

import libinteract as li

TEST_DIR = "../tests/files"
FILES_DIR = ".."
FFMASSES_DIR = os.path.join(FILES_DIR, "ff_masses")

TRAJ = os.path.join(TEST_DIR, "traj.xtc")
TOP = os.path.join(TEST_DIR, "top.tpr")
REF = os.path.join(TEST_DIR, "ref.pdb")
KBPATOMSFILE = os.path.join(FILES_DIR, "kbp_atomlist")
CGSFILE = os.path.join(FILES_DIR, "charged_groups.ini")
HBSFILE = os.path.join(FILES_DIR, "hydrogen_bonds.ini")
FFFILE = os.path.join(FILES_DIR, "ff.S050.bin64")


######################### LIBINTERACT TESTS ###########################

class TestLibinteract:

    #--------------------------- Fixtures ----------------------------#

    @pytest.fixture(scope = "class")
    def traj(self):
        return TRAJ

    @pytest.fixture(scope = "class")
    def top(self):
        return TOP

    @pytest.fixture(scope = "class")
    def ref(self):
        return REF

    @pytest.fixture(scope = "class")
    def kbpatomsfile(self):
        return KBPATOMSFILE

    @pytest.fixture(scope = "class")
    def cgsfile(self):
        return CGSFILE

    @pytest.fixture(scope = "class")
    def hbsfile(self):
        return HBSFILE

    @pytest.fixture(scope = "class")
    def uni(self, top, traj):
        return mda.Universe(top, traj)

    @pytest.fixture(scope = "class")
    def refuni(self, ref):
        return mda.Universe(ref, ref)

    @pytest.fixture(scope = "class")
    def kbp_reslist(self):
        return ["ALA", "ARG", "ASN", "ASP", "CYS", \
                "GLN", "GLU", "HIS", "ILE", "LEU", \
                "LYS", "MET", "PHE", "PRO", "SER", \
                "THR", "TRP", "TYR", "VAL"]

    @pytest.fixture(scope = "class")
    def hc_reslist(self):
        return ["ALA", "VAL", "LEU", "ILE", \
                "PHE", "PRO", "TRP", "MET"]

    @pytest.fixture(scope = "class")
    def cgs(self, cgsfile):
        return li.parse_cgsfile(cgsfile = cgsfile)

    @pytest.fixture(scope = "class")
    def hbs(self, hbsfile):
        return li.parse_hbsfile(hbsfile = hbsfile)

    @pytest.fixture(scope = "class")
    def kbpatoms(self, kbpatomsfile):
        return li.parse_kbpatomsfile(kbpatomsfile = kbpatomsfile)

    @pytest.fixture(scope = "class")
    def cgids(self, refuni, uni, cgs):
        return li.generate_cg_identifiers(refuni = refuni, \
                                          uni = uni, \
                                          cgs = cgs)

    @pytest.fixture(scope = "class")
    def scids(refuni, uni, hc_reslist):
        return li.generate_sc_identifiers(refuni = refuni, \
                                          uni = uni, \
                                          reslist = hc_reslist)

    #---------------------------- Tests ------------------------------#
    
    def test_parse_cgsfile(self, cgsfile):
        return li.parse_cgsfile(cgsfile = cgsfile)

    def test_parse_hbsfile(self, hbsfile):
        return li.parse_hbsfile(hbsfile = hbsfile)

    def test_parse_kbpatomsfile(self, kbpatomsfile):
        return li.parse_kbpatomsfile(kbpatomsfile = kbpatomsfile)

    def test_generate_cg_identifiers(self, refuni, uni, cgs):
        return li.generate_cg_identifiers(refuni = refuni, \
                                          uni = uni, \
                                          cgs = cgs)

    def test_generate_sc_identifiers(self, refuni, uni, hc_reslist):
        return li.generate_sc_identifiers(refuni = refuni, \
                                          uni = uni, \
                                          reslist = hc_reslist)

    def test_do_hbonds(self, refuni, uni):
        sel1 = "backbone or name H or name H1 or name H2 " \
               "or name H3 or name O1 or name O2 or name OXT"
        sel2 = f"protein and not ({sel1})"
        return li.do_hbonds(sel1 = sel1, \
                            sel2 = sel2, \
                            refuni = refuni, \
                            uni = uni, \
                            distance = 3.0, \
                            angle = 120.0, \
                            perco = 20.0, \
                            dofullmatrix = True, \
                            otherhbs = None, \
                            perresidue = True)

    """

    def test_calc_potential(distances, orderedsparses, refuni, uni):
        return li.calc_potential(distances = distances, \
                                 orderedsparses = orderedsparses, \
                                 refuni = refuni, \
                                 uni = uni, \
                                 kbT = 1.0)

    def test_do_potential(kbpatoms, kbp_reslist, potentialfile, \
                          uni, refuni):
        return li.do_potential(kbpatoms = kbpatoms, \
                               reslist = kbp_reslist, \
                               potentialfile = potentialfile, \
                               uni = uni, \
                               refuni = refuni, \
                               dofullmatrix = True, \
                               kbT = 1.0, \
                               seqdistco = 0)

    def test_calc_dist_matrix_hc(uni, scids):
        scidentifiers, scidxs, scselections = scids
        return li.calc_dist_matrix(uni = uni, \
                                   idxs = scidxs, \
                                   selections = scselections, \
                                   co = 5.0, 
                                   sb = False)

    def test_calc_dist_matrix_sb(uni, cgids):
        cgidentifiers, cgidxs, cgselections = cgids
        return li.calc_dist_matrix(uni = uni, \
                                   idxs = scidxs, \
                                   selections = cgselections, \
                                   co = 4.5, \
                                   sb = True, \
                                   sbmode = "diff")

    def test_assign_ff_masses(ffmasses, scids):
        scidentifiers, scidxs, scselections = scids
        return li.assign_ff_masses(ffmasses = ffmasses, \
                                   chosenselections = scselections)


    def test_calc_sc_fullmatrix(scids, percmat):
        scidentifiers, scidxs, scselections = scids
        return li.calc_sc_fullmatrix(identifiers = scidentifiers, \
                                     idxs = scids, \
                                     percmat = percmat, \
                                     perco = 20.0)

    def test_calc_cg_fullmatrix(cgids, percmat):
        cgidentifiers, cgidxs, cgselections = cgids
        return li.calc_cg_fullmatrix(identifiers = cgidentifiers, \
                                     idxs = cgidxs, \
                                     percmat = percmat, \
                                     perco = 20.0)

    def test_do_interact_hc(refuni, uni, ffmasses, hc_reslist):
        return li.do_interact(\
            generate_identifiers_func = li.generate_sc_identifiers, \
            refuni = refuni, \
            uni = uni, \
            co = 5.0, \
            perco = 20.0, \
            ffmasses = ffmasses, \
            calc_fullmatrix_func = li.calc_sc_fullmatrix, \
            sb = False, \
            reslist = hc_reslist)

    def test_do_interact_sb(refuni, uni, ffmasses, cgs):
        return li.do_interact(\
            generate_identifiers_func = li.generate_cg_identifiers, \
            refuni = refuni, \
            uni = uni, \
            co = 4.5, \
            perco = 20.0, \
            ffmasses = ffmasses, \
            calc_fullmatrix_func = li.calc_sc_fullmatrix, \
            sb = True, \
            sbmode = "diff", \
            cgs = cgs)
    """