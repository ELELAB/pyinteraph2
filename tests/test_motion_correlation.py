#!/usr/bin/env python

import numpy as np
import pytest
import sys
import os
import MDAnalysis as mda
from MDAnalysis.analysis import align

from pyinteraph.motion_correlation import calculate_dccm

@pytest.fixture
def simple_coords():
    return np.array([
        [[1, 1, 1], [2, 2, 2]],
        [[2, 2, 2], [3, 3, 3]],
        [[3, 3, 3], [4, 4, 4]]
    ])

@pytest.fixture
def fluct(simple_coords):
    mean_pos = simple_coords.mean(axis=0)
    return simple_coords - mean_pos

@pytest.fixture
def ref_dir(request):
    return os.path.join(request.fspath.dirname, 'data/single_chain')

@pytest.fixture
def traj_flucts(ref_dir):

    u = mda.Universe(os.path.join(ref_dir, 'sim.prot.A.pdb'),
                     os.path.join(ref_dir, 'traj.xtc'))
    ref = mda.Universe(os.path.join(ref_dir, 'sim.prot.A.pdb'))

    align.AlignTraj(u, ref, select="name CA", in_memory=True).run()

    atoms = u.select_atoms("name CA")
    coords = u.trajectory.timeseries(atoms, order="fac")    # so that the shape is (frames, atoms, coords)

    # Calculate fluctuations
    mean_pos = coords.mean(axis=0)
    return coords - mean_pos

@pytest.fixture
def reference_dccm(ref_dir):
    # load and remove first column which contains just row number
    return np.loadtxt(os.path.join(ref_dir, 'dccm_ca_aligned_to_ref.csv'), delimiter=',')[:,1:]

# Testing dccm_calc
def test_dccm_diagonal_one(fluct):
    """ Tests that all the diagonal values of the matrix are one """
    dccm = calculate_dccm(fluct)
    np.testing.assert_allclose(np.diag(dccm), np.ones(dccm.shape[0]))

def test_dccm_symmetry(fluct):
    """ Tests that the matrix are symmetrix """
    dccm = calculate_dccm(fluct)
    np.testing.assert_allclose(dccm, dccm.T)

def test_dccm_calculation(fluct):
    """ Tests that the dccm of simple_coords is as expected """
    dccm = calculate_dccm(fluct)
    expected = np.array([
        [1.0, 1.0],
        [1.0, 1.0]
    ])
    np.testing.assert_allclose(dccm, expected)

def test_reference_system(traj_flucts, reference_dccm):
    dccm = calculate_dccm(traj_flucts)
    print(reference_dccm)
    np.testing.assert_allclose(dccm, reference_dccm, atol=1e-04, rtol=1)
