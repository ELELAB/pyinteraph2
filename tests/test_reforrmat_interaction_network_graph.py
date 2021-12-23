import MDAnalysis
import networkx
import os
import numpy as np
import pytest
from numpy.testing import assert_array_equal

from pyinteraph.reformat_interaction_network_graph import ReformatDatGraph


@pytest.fixture
def change_test_dir(request):
    os.chdir('tests/data/single_chain')
    yield
    os.chdir(request.config.rootdir)


@pytest.fixture
def reference_structure():
    return 'sim.prot.A.pdb'


@pytest.fixture
def interaction_network_file():
    return 'hb-graph-filtered.dat'


@pytest.fixture
def empty_interaction_network_file():
    return 'xx.dat'


@pytest.fixture
def reformat_dat_graph(interaction_network_file, reference_structure, change_test_dir):
    return ReformatDatGraph(interaction_network_file, "output_name", reference_structure)


@pytest.fixture
def reformat_dat_graph_reference_structure_not_given(interaction_network_file, change_test_dir):
    return ReformatDatGraph(interaction_network_file, "output_name")


@pytest.fixture
def reformat_dat_graph_empty_interaction_network(empty_interaction_network_file, change_test_dir, reference_structure):
    return ReformatDatGraph(empty_interaction_network_file, "output_name", reference_structure)


@pytest.fixture
def interaction_network_initial_values():
    return np.array([ 0. ,  0. , 31.1, 79.5, 70.9,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,
        0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,
        0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,
        0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,
        0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,
        0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,
        0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,
        0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,
        0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,
        0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,
        0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,
        0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,
        0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,
        0. ,  0. ,  0. ,  0. ,  0. ,  0. ])


@pytest.fixture
def predefined_node_names():
    return ['A-2SER', 'A-3ARG', 'A-4ALA', 'A-5LYS', 'A-6ARG', 'A-7ILE', 'A-8MET', 'A-9LYS', 'A-10GLU', 'A-11ILE', 'A-12GLN', 'A-13ALA', 'A-14VAL', 'A-15LYS', 'A-16ASP', 'A-17ASP', 'A-18PRO', 'A-19ALA', 'A-20ALA', 'A-21HIS', 'A-22ILE', 'A-23THR', 'A-24LEU', 'A-25GLU', 'A-26PHE', 'A-27VAL', 'A-28SER', 'A-29GLU', 'A-30SER', 'A-31ASP', 'A-32ILE', 'A-33HIS', 'A-34HIS', 'A-35LEU', 'A-36LYS', 'A-37GLY', 'A-38THR', 'A-39PHE', 'A-40LEU', 'A-41GLY', 'A-42PRO', 'A-43PRO', 'A-44GLY', 'A-45THR', 'A-46PRO', 'A-47TYR', 'A-48GLU', 'A-49GLY', 'A-50GLY', 'A-51LYS', 'A-52PHE', 'A-53VAL', 'A-54VAL', 'A-55ASP', 'A-56ILE', 'A-57GLU', 'A-58VAL', 'A-59PRO', 'A-60MET', 'A-61GLU', 'A-62TYR', 'A-63PRO', 'A-64PHE', 'A-65LYS', 'A-66PRO', 'A-67PRO', 'A-68LYS', 'A-69MET', 'A-70GLN', 'A-71PHE', 'A-72ASP', 'A-73THR', 'A-74LYS', 'A-75VAL', 'A-76TYR', 'A-77HIS', 'A-78PRO', 'A-79ASN', 'A-80ILE', 'A-81SER', 'A-82SER', 'A-83VAL', 'A-84THR', 'A-85GLY', 'A-86ALA', 'A-87ILE', 'A-88CYS', 'A-89LEU', 'A-90ASP', 'A-91ILE', 'A-92LEU', 'A-93LYS', 'A-94ASN', 'A-95ALA', 'A-96TRP', 'A-97SER', 'A-98PRO', 'A-99VAL', 'A-100ILE', 'A-101THR', 'A-102LEU', 'A-103LYS', 'A-104SER', 'A-105ALA', 'A-106LEU', 'A-107ILE', 'A-108SER', 'A-109LEU', 'A-110GLN', 'A-111ALA', 'A-112LEU', 'A-113LEU', 'A-114GLN', 'A-115SER', 'A-116PRO', 'A-117GLU', 'A-118PRO', 'A-119ASN', 'A-120ASP', 'A-121PRO', 'A-122GLN', 'A-123ASP', 'A-124ALA', 'A-125GLU', 'A-126VAL', 'A-127ALA', 'A-128GLN', 'A-129HIS', 'A-130TYR', 'A-131LEU', 'A-132ARG', 'A-133ASP', 'A-134ARG', 'A-135GLU', 'A-136SER', 'A-137PHE', 'A-138ASN', 'A-139LYS', 'A-140THR', 'A-141ALA', 'A-142ALA', 'A-143LEU', 'A-144TRP', 'A-145THR', 'A-146ARG', 'A-147LEU', 'A-148TYR', 'A-149ALA', 'A-150SER']


def test_interaction_network(reformat_dat_graph):
    interaction_network = reformat_dat_graph.interaction_network
    assert isinstance(interaction_network, np.ndarray)


def test_interaction_network_empty_interaction_network(reformat_dat_graph_empty_interaction_network):
    with pytest.raises(SystemExit) as e:
        interaction_network = reformat_dat_graph_empty_interaction_network.interaction_network
    assert e.type == SystemExit
    assert e.value.code == 1


def test_interaction_network_compare_initial_values(reformat_dat_graph, interaction_network_initial_values):
    interaction_network = reformat_dat_graph.interaction_network
    assert_array_equal(interaction_network[0], interaction_network_initial_values)


def test_reference_structure(reformat_dat_graph):
    reference_structure = reformat_dat_graph.reference_structure
    assert isinstance(reference_structure, MDAnalysis.core.universe.Universe)
    assert len(reference_structure.residues) == 149
    assert reference_structure.residues[0].resname == "SER"
    assert reference_structure.residues[0].resid == 2


def test_node_names(reformat_dat_graph, predefined_node_names):
    node_names = reformat_dat_graph.node_names
    assert isinstance(node_names, list)
    assert node_names == predefined_node_names


def test_node_names_without_reference_structure(caplog, reformat_dat_graph_reference_structure_not_given, predefined_node_names):
    node_names = reformat_dat_graph_reference_structure_not_given.node_names
    assert isinstance(node_names, list)
    assert not node_names == predefined_node_names
    assert "Auto-generated numbers will be used to rename nodes" in caplog.text


def test_interaction_network_graph(reformat_dat_graph):
    interaction_network_graph = reformat_dat_graph.interaction_network_graph
    assert isinstance(interaction_network_graph, networkx.classes.graph.Graph)
    