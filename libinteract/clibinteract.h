/*
    PyInteraph, a software suite to analyze interactions and interaction network in structural ensembles.
    Copyright (C) 2013 Matteo Tiberti <matteo.tiberti@gmail.com>, Gaetano Invernizzi, Yuval Inbar, 
    Matteo Lambrughi, Gideon Schreiber,  Elena Papaleo <elena.papaleo@unimib.it> <elena.papaleo@bio.ku.dk>

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/


int trmI(int, int);
int sqmI(int, int, int);
double ed(double*, double*, int, int);

int potential_distances(double*, int, int, int, double*);
int triangular_distmatrix(double*, int, int, double, long*, double*);
int square_distmatrix(double*, double*, int, int, int, double, long*);
int triangular_mindist(double*, int, int, long*, double, long*);
int square_mindist(double*, double*, int, int, int, long*, long*, double, long*);

