#!/usr/bin/env python

#    PyInteraph, a software suite to analyze interactions and
#    interaction network in structural ensembles.
#    Copyright (C) 2022 Matteo Tiberti <matteo.tiberti@gmail.com>,
#                       Deniz Doğan <deenizzdogan@gmail.com>
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

import sys
import os

import argparse
import functools
import logging
import networkx as nx
import numpy as np
import MDAnalysis as mda
import warnings

class ArgumentParserFileExtensionValidation(argparse.FileType):
    parser = argparse.ArgumentParser()
    def __init__(self, valid_extensions, file_name):
        self.valid_extensions = valid_extensions
        self.file_name = file_name

    def validate_file_extension(self):
        given_extension = os.path.splitext(self.file_name)[1][1:]
        if given_extension not in self.valid_extensions:
            self.parser.error(f"Invalid file extension. Please provide a {self.valid_extensions} file")
        return self.file_name

warnings.filterwarnings("ignore")
logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s | %(message)s",
    datefmt="%Y/%m/%d %H:%M:%S"
)

logger = logging.getLogger(__name__)

class ReformatDatGraph:
    def __init__(self, interaction_network_file, output_name, reference_structure_file=None):
        self.reference_structure_file = reference_structure_file
        self.interaction_network_file = interaction_network_file
        self.output_name = output_name

    @property
    @functools.lru_cache()
    def interaction_network(self):
        try:
            interaction_network = np.loadtxt(self.interaction_network_file)
            return interaction_network
        except IOError:
            logger.error("Input network file not readable. Exiting...")
            sys.exit(1)
        except ValueError:
            logger.error("Input network file not in the right format. Exiting...")
        if interaction_network.shape[0] != interaction_network.shape[1]:
            logger.error("The input matrix is not square. Exiting...")
            exit(1)

    @property
    @functools.lru_cache()
    def reference_structure(self):
        if not self.reference_structure_file:
            return
        try:
            return mda.Universe(self.reference_structure_file)
        except:
            logger.warning("Reference structure not readable. Exiting...")
            sys.exit(1)
            return

    @property
    @functools.lru_cache()
    def node_names(self):
        if not self.reference_structure:
            logger.warning("Auto-generated numbers will be used as node names")
            return [str(i) for i in range(1, self.interaction_network.shape[0] + 1)]
        return list(map(lambda r: f"{r.segment.segid}-{r.resnum}{r.resname}", self.reference_structure.residues))

    @property
    @functools.lru_cache()
    def interaction_network_graph(self):
        if self.interaction_network is None:
            return
        try:
            interaction_network_graph = nx.Graph(self.interaction_network)
        except:
            logger.error("the input file doesn't contain a valid adjacency matrix")
            exit(1)
        if self.interaction_network.shape[0] != len(self.node_names):
            logger.error("the input network doesn't have the same number of residues as the input structure. Exiting...")
            exit(1)

        node_names = dict(zip(range(self.interaction_network.shape[0]), self.node_names))
        nx.relabel_nodes(interaction_network_graph, mapping=node_names, copy=False)
        return interaction_network_graph

    def graphml_formatted_interaction_network(self):
        nx.write_graphml(self.interaction_network_graph, f"{self.output_name}")


def main():
    parser = argparse.ArgumentParser(description="Format converting module to generate cytoscape compatible formatted "
                                                 "graph for PyInteraph.")
    parser.add_argument("-a", "--adj-matrix", dest="interaction_network_file", help=".dat file matrix",
                        type=lambda file_name: ArgumentParserFileExtensionValidation(("dat"),
                                                                                     file_name).validate_file_extension(),
                        required=True)
    parser.add_argument("-r", "--reference", dest="reference_structure_file", help="Reference PDB file",
                        type=lambda file_name: ArgumentParserFileExtensionValidation(("pdb"),
                                                                                     file_name).validate_file_extension(),
                        default=None)
    parser.add_argument("-o", "--output-name", dest="output_name", help="Specify graphml name", type=str,
                        default="graph.graphml")
    args = parser.parse_args()

    ReformatDatGraph(reference_structure_file=args.reference_structure_file,
                     interaction_network_file=args.interaction_network_file,
                     output_name=args.output_name).graphml_formatted_interaction_network()


if __name__ == "__main__":
    main()
