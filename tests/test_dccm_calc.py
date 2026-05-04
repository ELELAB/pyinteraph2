#!/usr/bin/env python

import numpy as np
import pytest
import sys
import os

from pyinteraph import dccm_calc

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

# Testing dccm_calc
def test_dccm_diagonal_one(fluct):
    """ Tests that all the diagonal values of the matrix are one """
    dccm = dccm_calc(fluct)
    np.testing.assert_allclose(np.diag(dccm), np.ones(dccm.shape[0]))

def test_dccm_symmetry(fluct):
    """ Tests that the matrix are symmetrix """
    dccm = dccm_calc(fluct)
    np.testing.assert_allclose(dccm, dccm.T)

def test_dccm_calculation(fluct):
    """ Tests that the dccm of simple_coords is as expected """
    dccm = dccm_calc(fluct)
    expected = np.array([
        [1.0, 1.0],
        [1.0, 1.0]
    ])
    np.testing.assert_allclose(dccm, expected)
