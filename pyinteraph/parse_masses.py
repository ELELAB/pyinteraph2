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

import json
import sys
import argparse

def parse_gmxff(atp,rtp,doc,outfile):
    wrongheaders=["[ bondedtypes ]\n","[ atoms ]\n","[ bonds ]\n","[ impropers ]\n","[ dihedrals ]\n"]

    atomtypes={}
    residues={}
    name = ""

    _doc = open(doc,'r')
    doclines = _doc.readlines()
    name = doclines[0].strip()

    _atp = open(atp,'r')
    for line in _atp:
        if line.startswith(";"):
            continue
        pline = line.partition(";")[0]
        atomtypes[pline.split()[0]] = float(line.split()[1])
  
        
    _rtp = open(rtp,'r')
    rtplines = _rtp.readlines()
    for l in range(len(rtplines)):
        if line.startswith(";"):
            continue 
        rtplines[l] = rtplines[l].partition(";")[0]
    for l in range(len(rtplines)):
        if rtplines[l].startswith("[ ") and rtplines[l] not in wrongheaders:
            k=l+2
            resname = rtplines[l].strip().rstrip(" ]").lstrip(" [")
            residues[resname] = {}
            while True:
                if rtplines[k].startswith(" [") or rtplines[k].startswith("\n"):
                    break
                tmp = rtplines[k].rstrip().split()
                try:
                    residues[resname][tmp[0]] = atomtypes[tmp[1]]
                except:
                    pass
                k+=1
            if len(residues[resname]) == 0:
                residues.pop(resname)
    _outfile = open(outfile,'w')
    json.dump([name,residues],_outfile)
    _outfile.close

def main():

    parser = argparse.ArgumentParser(description='Create a PyInteraph mass file from a GROMACS-defined force-field.')

    parser.add_argument('-a', '--atp',dest='atp',help="Gromacs force-field atp file", required=True)
    parser.add_argument('-r', '--rtp',dest='rtp',help="Gromacs force-field rtp file", required=True)
    parser.add_argument('-d', '--doc',dest='doc',help="Gromacs force-field dtp file", required=True)
    parser.add_argument('-o', '--out',dest='out',help="Output file", required=True)

    args = parser.parse_args()
    
    try:
        parse_gmxff(args.atp, args.rtp, args.doc, args.out)
    except IOError:
        print("An error occurred parsing the input files; exiting..")

