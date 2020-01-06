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

vinfo = sys.version_info
if vinfo[0] < 2 or (vinfo[0] == 2 and vinfo[1] < 7):
    errstr = \
        "Your Python version is {:s}, but only " \
        "versions >= 2.7 are supported."
    log.error(errstr.format(sys.version))
    exit(1)

import argparse

import numpy as np
from scipy.optimize import curve_fit
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
import networkx as nx


########################## HELPER FUNCTIONS ###########################

# Parametric sigmoid function for fitting
def sigmoid(x, x0, k, m, n): 
    y = m / (1 + np.exp(k*(x-x0))) + n
    return y

# Parametric analytic second derivative of sigmoid 
def seconddevsigmoid(x, x0, k, l, m): 
    y = \
        (k**2 * l * np.exp(k*(x+x0)) * (np.exp(k*x)-np.exp(k*x0)))  \
        / (np.exp(k*x0) + np.exp(k*x))**3    
    
    return y

# Load matrix
def load_matrix(fname):
    try:
        matrix = np.loadtxt(fname)
        return matrix
    except:
        # catch numpy errors and include the traceback when
        # log
        logstr = "Could not open file {:s}, or file in wrong format"
        log.error(logstr.format(fname, exc_info = True))
        exit(1)       


# Process matrices
def process_matrices(fnames) 
    matrices = [load_matrix(fname) for fname in fnames]
    shapes = [matrix.shape for matrix in matrices]

    if len(set(shapes)) != 1:
        log.error("Matrices do not have the same shape.")
        exit(1)

    for fname, matrix in zip(fnames, matrices):
        if matrix.shape[0] != matrix.shape[1]:
            logstr = "Matrix {:s} is not square"
            log.error(logstr.format(fname))
            exit(1)
        if not np.allclose(matrix, matrix.T):
            logstr = "Matrix {:s} is not symmetric"
            log.error(logstr.format(fname))
            exit(1)

        # all diagonal elements must be zero
        np.fill_diagonal(matrix, 0.0)

    return matrices

# Get maximum cluster sizes
def get_maxclustsizes(matrices, interval)
    maxclustsizes = []
    for val in interval:
        # boolean matrices indicating where the original
        # matrices exceed val       
        boolmatrices = [matrix > val for matrix in matrices]
        # there will be only one boolean matrix if only one
        # input matrix was provided
        allmatrix = boolmatrices[0]
        if len(boolmatrices) > 1:
            # in case more there was more than one input matrix,
            # the final matrix will be a matrix resulting from an
            # element-wise logical OR applied to all matrices
            for i in range(1, len(boolmatrices)):
                allmatrix = np.logical_or(allmatrix, boolmatrices[i])

        # build a graph from the final boolean matrix
        G = nx.Graph(allmatrix)
        # get the maximum cluster size from the graph
        maxclustsizes.append(\
            len(max(list(\
                nx.algorithms.components.connected_components(G)), \
            key = len)))

    return maxclustsizes


# Perform curve fitting
def perform_fitting(f, xdata, ydata, maxfev, p0):
    # args will be None unless the fitting completes successfully
    args = None
    try:
        popt, pcov = \
            curve_fit(f = f, \
                      xdata = xdata, \
                      ydata = ydata, \
                      maxfev = maxfev, \
                      p0 = p0)
            
        args = (popt[0], popt[1], popt[2], popt[3])
        
    except ValueError:
        # as per scipy.optimize.curve_fit documentation
        log.error(\
            "Please check input data and options provided for " \
            "the fitting", \
            exc_info = True)

    except RuntimeError:
        # as per scipy.optimize.curve_fit documentation
        log.error(\
            "Could not complete fitting since the least-squares ", \
            "minimization failed", \
            exc_info = True)

    except:
        # something else happened (should not happen)
        log.error(\
            "Could not complete fitting", \
            exc_info = True)

    return args


# Find point of inflexion
def find_flex(func, x0, args, maxfev):
    # flex will be None unless the calculation completes successfully
    flex = None
    try:
        flex = fsolve(func = seconddevsigmoid, \
                      x0 = x0, \
                      args = args, \
                      maxfev = maxfev)    
    except:
        # inform the user that something went wrong during
        # the calculation and include traceback
        log.error(\
            "Could not complete the calculation", \
            exc_info = True)

    return flex


# Plot
def perform_plotting(x, \
                     y, \
                     lower, \
                     upper, \
                     out_plot, \
                     args = None, \
                     popt = None, \
                     flex = None, \
                     func_sigmoid = None):

    plt.plot(x = x, y = y, fmt = "o")
    plt.xlim((lower, upper))
    plt.xlabel("$p_{min}$")
    plt.ylabel("Size of the biggest cluster")
        
    if args is not None and popt is not None:
        xplot = np.linspace(max(x), min(x))
        yplot = sigmoid(xplot, *popt)
        plt.plot(x = xplot, y = yplot, label = "Fitting")
             
    if flex is not None and func_sigmoid is not None:
        plt.plot(x = flex, \
                 y = func_sigmoid(flex, *popt),\
                 fmt = "o", \
                     label = "Critical value", \
                 color = "red")
            
        plt.axvline(x = flex)

    plt.legend(loc = "best")
    plt.savefig(out_plot)


# Write clusters to an output file
def write_clusters(out_clusters, x):
    try:
        fh = open(out_clusters, "w")
    except:
        log.error(\
            "Could not write clusters file.", \
            exc_info = True)
        exit(1)
        
    fh.write("P_min\tSize of biggest cluster\n")
    for i, xi in enumerate(x):
        fh.write("{:.3f}\t{:d}\n".format(xi, y[i]))
        
    fh.close()


# Write matrices to a .dat file (logical OR applied if 
# multiple input matrices provided)
def write_dat(matrices, \
              matrix_filter, \
              out_dat, \
              weights = None):
    
    if len(matrices) == 1:
        # where the mask is True, the corresponding value 
        # in the array is invalid    
        mask = (matrices[0] <= matrix_filter)
        out_matrix = np.ma.masked_array(data = matrices[0], \
                                        mask = mask, \
                                        fill_value = 0.0).filled()
    else:
        # boolean matrices indicating where the original
        # matrices exceed val  
        boolmatrices = [matrix > matrix_filter for matrix in matrices]
        # in case more there was more than one input matrix,
        # the final matrix will be a matrix resulting from an
        # element-wise logical OR applied to all matrices
        out_matrix = np.logical_or.reduce(boolmatrices)
        
    if weights is not None:
        # if a matrix of weights was provided
        try:
            weights_matrix = np.loadtxt(weights)
        except:
            log.error(\
                "Could not read weights matrix.", \
                exc_info = True)
            exit(1)
             
        if weights_matrix.shape != out_matrix.shape:
            log.error(\
                "Output and weight matrix have different shapes.")
            exit(1)

        out_matrix = np.ma.masked_array(data = weights_matrix, \
                                        mask = out_matrix > 0.0, \
                                        fill_value = 0.0).filled()

    np.savetxt(out_dat, out_matrix, fmt = "%3.2f")



if __name__ == "__main__":

    ########################## ARGUMENT PARSER ############################

    description = "Find persistence critical value"
    parser = argparse.ArgumentParser(description = description)

    d_helpstr = \
        ".dat file matrices (multiple: -d file.dat -d file2.dat ...)"
    parser.add_argument("-d", "--input-dat", \
                        dest = "datfiles", \
                        help = d_helpstr, \
                        action = "append", \
                        default = None)

    o_helpstr = "Output .dat file matrix"
    parser.add_argument("-o", "--output-dat", \
                        dest = "out_dat", \
                        help = o_helpstr, \
                        action = "store", \
                        type = str, \
                        default = None)

    c_helpstr = "Output clusters file"
    parser.add_argument("-c", "--output-clusters", \
                        dest = "out_clusters", \
                        help = c_helpstr, \
                        action = "store", \
                        type = str, \
                        default = None)

    t_default = 0.0
    t_helpstr = \
        "Filter input matrices according to this threshold (default: {:f})"
    parser.add_argument("-t", "--filter-threshold", \
                        dest = "filter", \
                        help = t_helpstr.format(t_default), \
                        type = float, \
                        default = t_default)

    p_helpstr = "Name of the file where to plot clusters"
    parser.add_argument("-p", "--plot", \
                        dest = "out_plot", \
                        help = p_helpstr, \
                        default = None)

    f_helpstr = \
        "Try to fit the plot to a sigmoid y = m / (1 + exp(k*(x-x0))) + n"
    parser.add_argument("-f", "--fit", \
                        dest = "do_fit", \
                        help = f_helpstr, \
                        action = "store_true")

    u_default = 100.0
    u_helpstr = \
        "Maxium range value to be considered (default: {:f})"
    parser.add_argument("-u", "--range-upper", \
                        dest = "upper", \
                        help = u_helpstr.format(u_default), \
                        type = float, \
                        default = u_default)

    l_default = 0.0
    l_helpstr = "Minimum range value to be considered (default: {:f})"
    parser.add_argument("-l", "--range-lower", \
                        dest = "lower", \
                        help = l_helpstr.format(l_default), \
                        type = float, \
                        default = l_default)

    s_default = 5.0
    s_helpstr = \
        "Step range value to be considered (default: {:f})"
    parser.add_argument("-s", "--range-step", \
                        dest = "step", \
                        help = s_helpstr.format(s_default), \
                        type = float, \
                        default = s_default)

    w_helpstr = \
        "Use values in this matrix as weights for the output " \
        "adjacency matrix"
    parser.add_argument("-w", "--weight-matrix", \
                        dest = "weights", \
                        help = w_helpstr, \
                        type = str, \
                        default = None)

    x_default = 20.0
    x_helpstr = \
        "Starting x0 parameter for sigmoid fitting (default: {:f})"
    parser.add_argument("-x", "--x0", \
                        dest = "x0", \
                        help = x_helpstr.format(x_default), \
                        type = float, \
                        default = x_default)

    k_default = 2.0
    k_helpstr = \
        "Starting k parameter for sigmoid fitting (default: {:f})"
    parser.add_argument("-k", \
                        dest = "k", \
                        help = k_helpstr.format(k_default),
                        type = float, \
                        default = k_default)

    m_default = 20.0
    m_helpstr = \
        "Starting m parameter for sigmoid fitting (default: {:f})"
    parser.add_argument("-m", \
                        dest = "m", \
                        help = m_helpstr.format(m_default), \
                        type = float, \
                        default = m_default)

    n_default = 10.0
    n_helpstr = \
        "Starting n parameter for sigmoid fitting (default: {:f})"
    parser.add_argument("-n", \
                        dest = "n", \
                        help = n_helpstr.format(n_default), \
                        type = float, \
                        default = n_default)

    options = parser.parse_args()


    ########################### CHECK INPUTS ##############################

    if options.datfiles is None:
        log.error("Input file(s) must be provided.")
        exit(1)

    if options.upper <= options.lower:
        log.error("Maximum range value must be higher than minimum.")
        exit(1)

    if options.step > (options.upper - options.lower):
        logstr = \
            "Step value must be lower than or equal to " \
            "[upper_value - lower_value]"
        log.error(logstr)
        exit(1)

    matrices = process_matrices(fnames = options.datfiles)


    ####################### MAXIMUM CLUSTER SIZES #########################

    interval = np.arange(options.lower, options.upper, options.step)
    maxclustsizes = get_maxclustsizes(matrices = matrices, \
                                      interval = interval)


    ############################## FITTING ################################

    x = interval
    y = maxclustsizes
    args = None
    flex = None

    if options.do_fit:
        logstr = \
            '"""\nFitting data points to functional form: ' \
            'y = m / (1 + exp(k*(x-x0))) + n\nStarting parameters:\n' \
            'x0 = {:3.2f}\nk =  {:3.2f}\nm =  {:3.2f}\nn =  {:3.2f}\n"""'
        log.info(\
            logstr.format(options.x0, options.k, options.m, options.n))

        # args will be None unless the fitting completes successfully
        args = perform_fitting(f = sigmoid, \
                               xdata = x, \
                               ydata = y, \
                               maxfev = 100000, \
                               p0 = (options.x0, options.k, \
                                     options.m, options.n))


        ####################### SECOND DERIVATIVE #########################
        
        if args is not None:
            logstr = \
                '"""\nDone! Calculated parameters:\nx0 = {:3.5f}\n' \
                'k  = {:3.5f}\nl  = {:3.5f}\nm  = {:3.5f}\n"""'
            log.info(logstr.format(*args))

            log.info(\
                "Looking for central point of inflection (f''(x) = 0) ...")

            solvestart = options.x0
            log.info("Starting from: {:3.5f} ...".format(solvestart))

            # flex will be None unless the calculation completes
            # successfully
            flex = find_flex(func = seconddevsigmoid, \
                             x0 = solvestart, \
                             args = args, \
                             maxfev = 5000)

            if flex is not None:
                log.info("Flex at {:3.2f}".format(flex))
            else:
                log.info("No flex found")


    ############################## PLOTTING ###############################

    if options.out_plot is not None:
        perform_plotting(x = x, \
                         y = y, \
                         lower = options.lower, \
                         upper = options.upper, \
                         out_plot = options.out_plot, \
                         args = args, \
                         popt = popt, \
                         flex = flex, \
                         func_sigmoid = sigmoid)


    ########################### OUTPUT CLUSTERS ###########################

    if options.out_clusters is not None:
        write_clusters(out_clusters = options.out_clusters, \
                       x = x)


    ############################# OUTPUT DAT ##############################

    if options.out_dat is not None:
        write_dat(matrices = matrices, \
                  matrix_filter = options.filter, \
                  out_dat = options.out_dat, \
                  weights = options.weights)
