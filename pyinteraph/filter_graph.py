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
import sys
import logging as log

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from scipy.optimize import curve_fit
from scipy.optimize import fsolve


########################## HELPER FUNCTIONS ###########################

def sigmoid(x, x0, k, m, n):
    """Parametric sigmoid function for fitting.""" 
    y = m / (1 + np.exp(k*(x-x0))) + n
    return y
 

def seconddevsigmoid(x, x0, k, l, m):
    """Parametric analytic second derivative of sigmoid."""
    y = \
        (k**2 * l * np.exp(k*(x+x0)) * (np.exp(k*x) - np.exp(k*x0))) \
        / (np.exp(k*x0) + np.exp(k*x))**3     
    return y


def process_matrices(fnames):
    """Load and process matrix files. """

    # empty list to store matrices
    matrices = []
    for fname in fnames:
        try:
            matrices.append(np.loadtxt(fname))
        except:
            # catch numpy errors and include the traceback when logging
            errstr = \
                f"Could not open file {fname}, or file in wrong format."
            raise IOError(errstr)
    # get matrix shapes
    shapes = [matrix.shape for matrix in matrices]
    # all matrices must have the same shape
    if len(set(shapes)) != 1:
        raise ValueError("Matrices do not have the same shape.")
    # check if the matrix are square and symmetric
    for fname, matrix in zip(fnames, matrices):
        if matrix.shape[0] != matrix.shape[1]:
            raise ValueError(f"Matrix {fname} is not square.")
        if not np.allclose(matrix, matrix.T):
            raise ValueError(f"Matrix {fname} is not symmetric.")
        # all diagonal elements must be zero
        np.fill_diagonal(matrix, 0.0)
    # return checked matrices
    return matrices


def get_maxclustsizes(matrices, interval):
    """Get maximum cluster sizes."""
    
    # empty list to store maximum cluster sizes
    maxclustsizes = []
    # for each value in the interval
    for val in interval:
        # boolean matrices indicating where the original
        # matrices exceed val       
        boolmatrices = [matrix > val for matrix in matrices]
        # in case more there was more than one input matrix,
        # the final matrix will be a matrix resulting from an
        # element-wise logical OR applied to all matrices
        allmatrix = np.logical_or.reduce(boolmatrices)
        # build a graph from the final boolean matrix
        G = nx.Graph(allmatrix)
        # get the maximum cluster size from the graph
        maxclustsizes.append(\
            len(max(list(\
                nx.algorithms.components.connected_components(G)), \
            key = len)))
    # return the list of maximum cluster sizes
    return maxclustsizes


def perform_fitting(f, xdata, ydata, maxfev, p0):
    """Perform curve fitting."""
    
    # args will be None unless the fitting completes successfully
    args = None
    try:
        # try to perform the curve fitting
        popt, pcov = \
            curve_fit(f = f, \
                      xdata = xdata,
                      ydata = ydata,
                      maxfev = maxfev,
                      p0 = p0)
        # results of the fitting
        args = (popt[0], popt[1], popt[2], popt[3])
    # invalid inputs or options
    except ValueError:
        # as per scipy.optimize.curve_fit documentation
        log.error(\
            "Please check input data and options provided for " \
            "the fitting", \
            exc_info = True)
    # runtime error during optimization
    except RuntimeError:
        # as per scipy.optimize.curve_fit documentation
        log.error(\
            "Could not complete fitting since the least-squares ", \
            "minimization failed", \
            exc_info = True)
    # other errors
    except:
        # something else happened (should not happen)
        log.error(\
            "Could not complete fitting", \
            exc_info = True)
    # return results of the fitting or None if the fitting
    # could not complete
    return args


def find_flex(func, x0, args, maxfev):
    """Find the point of inflection."""

    # flex will be None unless the calculation completes successfully
    return fsolve(func = seconddevsigmoid,
                  x0 = x0,
                  args = args,
                  maxfev = maxfev,
                  full_output = True)


def perform_plotting(x,
                     y,
                     lower,
                     upper,
                     out_plot,
                     args = None,
                     flex = None,
                     func_sigmoid = None):
    """Plot the dependency between the persistence
    cut-off and the size of the biggest cluster."""

    # plot as dots
    plt.plot(x, y, "o")
    # set X-axis limit
    plt.xlim((lower, upper))
    # set axes labels
    plt.xlabel("$p_{min}$")
    plt.ylabel("Size of the biggest cluster")
    # plot sigmoidal fitting
    if args is not None:
        xplot = np.linspace(max(x), min(x))
        yplot = func_sigmoid(xplot, *args)
        plt.plot(xplot, yplot, label = "Fitting")
        # plot the inflection point, if found
        if flex is not None:
            plt.plot(flex,
                     func_sigmoid(flex, *args),
                     "o",
                     label = "Critical value",
                     color = "red")
            # plot a vertical line corresponding
            # to the inflection point
            plt.axvline(x = flex)
    # plot the legend
    plt.legend(loc = "best")
    # save the figure
    plt.savefig(out_plot)


def write_clusters(out_clusters, interval, maxclustsizes):
    """Write clusters to an output file."""
    
    try:
        fh = open(out_clusters, "w")
    except:
        raise IOError("Could not write clusters file.")  
    with fh:
        fh.write("P_min\tSize of biggest cluster\n")
        for pmin, maxclustsize in zip(interval, maxclustsizes):
            fh.write("{:.3f}\t{:d}\n".format(pmin, maxclustsize))


def write_dat(matrices,
              matrix_filter,
              out_dat,
              weights = None):

    """Write matrices to a .dat file (logical OR applied if
    multiple input matrices provided)"""
    
    if len(matrices) == 1:
        # where the mask is True, the corresponding value 
        # in the array is invalid    
        mask = (matrices[0] <= matrix_filter)
        out_matrix = np.ma.masked_array(data = matrices[0],
                                        mask = mask,
                                        fill_value = 0.0).filled()
    else:
        # boolean matrices indicating where the original
        # matrices exceed val  
        boolmatrices = [matrix > matrix_filter for matrix in matrices]
        # in case more there was more than one input matrix,
        # the final matrix will be a matrix resulting from an
        # element-wise logical OR applied to all matrices
        out_matrix = np.logical_or.reduce(boolmatrices)
    # if a matrix of weights was provided 
    if weights is not None:
        # try to open the matrix file
        try:
            weights_matrix = np.loadtxt(weights)
        except:
            raise IOError("Could not read weights matrix.")
        # check the shape of the matrix of weights
        if weights_matrix.shape != out_matrix.shape:
            raise ValueError("Output and weight matrix "
                             "have different shapes.")
        # update out matrix
        out_matrix = np.ma.masked_array(data = weights_matrix,
                                        mask = out_matrix > 0.0,
                                        fill_value = 0.0).filled()
    # save the output matrix
    np.savetxt(out_dat, out_matrix, fmt = "%3.2f")




def main():
    ########################## ARGUMENT PARSER ############################

    description = "Find persistence critical value"
    parser = argparse.ArgumentParser(description = description)

    d_helpstr = \
        ".dat file matrices (multiple: -d file.dat -d file2.dat ...)"
    parser.add_argument("-d", "--input-dat",
                        dest = "datfiles",
                        help = d_helpstr,
                        action = "append",
                        default = None)

    o_helpstr = "Output .dat file matrix"
    parser.add_argument("-o", "--output-dat",
                        dest = "out_dat",
                        help = o_helpstr,
                        action = "store",
                        type = str,
                        default = None)

    c_helpstr = "Output clusters file"
    parser.add_argument("-c", "--output-clusters",
                        dest = "out_clusters",
                        help = c_helpstr,
                        action = "store",
                        type = str,
                        default = None)

    t_default = 0.0
    t_helpstr = \
        "Filter input matrices according to this threshold (default: {:f})"
    parser.add_argument("-t", "--filter-threshold",
                        dest = "filter",
                        help = t_helpstr.format(t_default),
                        type = float,
                        default = t_default)

    p_helpstr = "Name of the file where to plot clusters"
    parser.add_argument("-p", "--plot",
                        dest = "out_plot",
                        help = p_helpstr,
                        default = None)

    f_helpstr = \
        "Try to fit the plot to a sigmoid y = m / (1 + exp(k*(x-x0))) + n"
    parser.add_argument("-f", "--fit",
                        dest = "do_fit",
                        help = f_helpstr,
                        action = "store_true")

    u_default = 100.0
    u_helpstr = \
        "Maxium range value to be considered (default: {:f})"
    parser.add_argument("-u", "--range-upper",
                        dest = "upper",
                        help = u_helpstr.format(u_default),
                        type = float,
                        default = u_default)

    l_default = 0.0
    l_helpstr = "Minimum range value to be considered (default: {:f})"
    parser.add_argument("-l", "--range-lower",
                        dest = "lower",
                        help = l_helpstr.format(l_default),
                        type = float,
                        default = l_default)

    s_default = 5.0
    s_helpstr = \
        "Step range value to be considered (default: {:f})"
    parser.add_argument("-s", "--range-step",
                        dest = "step",
                        help = s_helpstr.format(s_default),
                        type = float,
                        default = s_default)

    w_helpstr = \
        "Use values in this matrix as weights for the output " \
        "adjacency matrix"
    parser.add_argument("-w", "--weight-matrix",
                        dest = "weights",
                        help = w_helpstr,
                        type = str,
                        default = None)

    x_default = 20.0
    x_helpstr = \
        "Starting x0 parameter for sigmoid fitting (default: {:f})"
    parser.add_argument("-x", "--x0",
                        dest = "x0",
                        help = x_helpstr.format(x_default),
                        type = float,
                        default = x_default)

    k_default = 2.0
    k_helpstr = \
        "Starting k parameter for sigmoid fitting (default: {:f})"
    parser.add_argument("-k",
                        dest = "k",
                        help = k_helpstr.format(k_default),
                        type = float,
                        default = k_default)

    m_default = 20.0
    m_helpstr = \
        "Starting m parameter for sigmoid fitting (default: {:f})"
    parser.add_argument("-m",
                        dest = "m",
                        help = m_helpstr.format(m_default),
                        type = float,
                        default = m_default)

    n_default = 10.0
    n_helpstr = \
        "Starting n parameter for sigmoid fitting (default: {:f})"
    parser.add_argument("-n",
                        dest = "n",
                        help = n_helpstr.format(n_default),
                        type = float,
                        default = n_default)

    options = parser.parse_args()


    ########################### CHECK INPUTS ##############################

    # check matrices
    if not options.datfiles:
        log.error("Input file(s) must be provided.")
        exit(1)
    # check range
    if options.upper <= options.lower:
        log.error("Maximum range value must be higher than minimum.")
        exit(1)
    # check step
    if options.step > (options.upper - options.lower):
        logstr = \
            "Step value must be lower than or equal to " \
            "[upper_value - lower_value]."
        log.error(logstr)
        exit(1)
    # process matrices
    matrices = process_matrices(fnames = options.datfiles)


    ####################### MAXIMUM CLUSTER SIZES #########################

    # set the interval of persistence cut-offs
    interval = np.arange(options.lower, options.upper, options.step)
    # find the maximum cluster size at each cut-off
    maxclustsizes = get_maxclustsizes(matrices = matrices,
                                      interval = interval)


    ############################## FITTING ################################

    args = None
    if options.do_fit:
        logstr = \
            '"""\nFitting data points to functional form: ' \
            'y = m / (1 + exp(k*(x-x0))) + n\nStarting parameters:\n' \
            'x0 = {:3.2f}\nk =  {:3.2f}\nm =  {:3.2f}\nn =  {:3.2f}\n"""'
        log.info(\
            logstr.format(options.x0, options.k, options.m, options.n))
        # args will be None unless the fitting completes successfully
        args = perform_fitting(f = sigmoid,
                               xdata = interval,
                               ydata = maxclustsizes,
                               maxfev = 100000,
                               p0 = (options.x0, options.k,
                                     options.m, options.n))

        ####################### SECOND DERIVATIVE #########################
        
        if args is not None:
            logstr = \
                '"""\nDone! Calculated parameters:\nx0 = {:3.5f}\n' \
                'k  = {:3.5f}\nl  = {:3.5f}\n"""'
            log.info(logstr.format(*args))
            log.info(\
                "Looking for central point of inflection (f''(x) = 0) ...")

            solvestart = options.x0
            log.info("Starting from: {:3.5f} ...".format(solvestart))
            # find the inflection point
            flex, infodict, ier, mesg = \
                find_flex(func = seconddevsigmoid,
                          x0 = solvestart,
                          args = args,
                          maxfev = 5000)
            # log the result of the flex search
            if ier == 1:
                log.info("Inflection point at {:3.2f}.".format(flex))
            else:
                log.info("No inflection point found. " \
                         "Reason: {:s}. Last iteration value: " \
                         " {:3.2f}".format(mesg, flex[0]))
    else:
        args = None
        flex = None
        func_sigmoid = None

    ############################## PLOTTING ###############################

    if options.out_plot:
        perform_plotting(x = interval,
                         y = maxclustsizes,
                         lower = options.lower,
                         upper = options.upper,
                         out_plot = options.out_plot,
                         args = args,
                         flex = flex,
                         func_sigmoid = sigmoid)


    ########################### OUTPUT CLUSTERS ###########################

    if options.out_clusters:
        write_clusters(out_clusters = options.out_clusters,
                       interval = interval,
                       maxclustsizes = maxclustsizes)


    ############################# OUTPUT DAT ##############################

    if options.out_dat:
        write_dat(matrices = matrices,
                  matrix_filter = options.filter,
                  out_dat = options.out_dat,
                  weights = options.weights)


if __name__ == "__main__":
    main()

