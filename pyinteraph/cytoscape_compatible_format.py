import sys

import argparse
import functools
import logging
import networkx as nx
import numpy as np
import MDAnalysis as mda
import warnings


from pyinteraph.file_validation import ArgumentParserFileExtensionValidation

warnings.filterwarnings("ignore")
logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s | %(message)s",
    datefmt="%Y/%m/%d %H:%M:%S"
)

logger = logging.getLogger(__name__)


class ReformatDatGraph:
    def __init__(self, interaction_network_file, output_name, reference_structure_file):
        self.reference_structure_file = reference_structure_file
        self.interaction_network_file = interaction_network_file
        self.output_name = output_name

    @property
    @functools.lru_cache()
    def interaction_network(self):
        try:
            interaction_network = np.loadtxt(self.interaction_network_file)
            return interaction_network
        except:
            sys.exit(1)

    @property
    @functools.lru_cache()
    def reference_structure(self):
        if not self.reference_structure_file:
            return
        try:
            return mda.Universe(self.reference_structure_file)
        except:
            logger.warning("Reference structure not readable.")
            return

    @property
    @functools.lru_cache()
    def node_names(self):
        if not self.reference_structure:
            logger.warning("Auto-generated numbers will be used to rename nodes.")
            return [str(i) for i in range(1, self.interaction_network.shape[0] + 1)]
        return list(map(lambda r: f"{r.segment.segid}-{r.resnum}{r.resname}", self.reference_structure.residues))

    @property
    def interaction_network_graph(self):
        try:
            uploaded_interaction_network_graph = nx.Graph(self.interaction_network)
            node_names = dict(zip(range(self.interaction_network.shape[0]), self.node_names))
            nx.relabel_nodes(uploaded_interaction_network_graph, mapping=node_names, copy=False)
            return uploaded_interaction_network_graph
        except:
            logger.error("Reformatted interaction network graph not generated. "
                         "Please check the content of the interaction network file")
            sys.exit(1)

    @property
    def graphml_formatted_interaction_network(self):
        return nx.write_graphml(self.interaction_network_graph, f"{self.output_name}.graphml")


def main():
    parser = argparse.ArgumentParser(description="Format converting module to generate cytoscape compatible formatted "
                                                 "graph for PyInteraph.")
    parser.add_argument("-i", "--interaction_network", dest="interaction_network_file", help=".dat file matrix.",
                        type=lambda file_name: ArgumentParserFileExtensionValidation(("dat"),
                                                                                     file_name).validate_file_extension(),
                        required=True)
    parser.add_argument("-r", "--reference_structure", dest="reference_structure_file", help="Reference PDB file.",
                        type=lambda file_name: ArgumentParserFileExtensionValidation(("pdb"),
                                                                                     file_name).validate_file_extension(),
                        default=None)
    parser.add_argument("-o", "--output-name", dest="output_name", help="Specify graphml name", type=str,
                        default="reformatted_graph")
    args = parser.parse_args()

    ReformatDatGraph(reference_structure_file=args.reference_structure_file,
                     interaction_network_file=args.interaction_network_file,
                     output_name=args.output_name).graphml_formatted_interaction_network


if __name__ == "__main__":
    main()
