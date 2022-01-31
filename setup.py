from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext as _build_ext
import numpy

libinteract = \
      Extension("libinteract.innerloops",
                ["libinteract/innerloops.pyx",
                 "libinteract/clibinteract.c"], \
                include_dirs = [numpy.get_include()])

name = "pyinteraph"

url = "https://www.github.com/ELELAB/pyinteraph"

author = "Matteo Tiberti, Gaetano Invernizzi, " \
         "Yuval Inbar, Matteo Lambrughi, " \
         "Gideon Schreiber, Elena Papaleo"

author_email = "matteo.tiberti@gmail.com"

version = "1.1"

description = "Compute interatomic interactions in protein ensembles"

ext_modules = [libinteract]

package_data = {"pyinteraph" : \
                  ["charged_groups.ini",
                   "hydrogen_bonds.ini",
                   "kbp_atomlist",
                   "charged_groups.ini",
                   "hydrogen_bonds.ini",
                   "normalization_factors.ini",
                   "ff.S050.bin64",
                   "ff_masses/*"]}

package_dir = {"libinteract" : "libinteract",
               "pyinteraph" : "pyinteraph"}

packages = ["libinteract", "pyinteraph"]

entry_points = {"console_scripts" : [\
                  "pyinteraph = pyinteraph.main:main",
                  "graph_analysis = pyinteraph.graph_analysis:main",
                  "filter_graph = pyinteraph.filter_graph:main",
                  "parse_masses = pyinteraph.parse_masses:main",
                  "path_analysis = pyinteraph.path_analysis:main",
                  "centrality_analysis = pyinteraph.centrality_analysis:main",
                  "dat2graphml = pyinteraph.dat2graphml:main"]}

install_requires = ["cython",
                    "biopython",
                    "MDAnalysis==1.1.1",
                    "numpy",
                    "matplotlib",
                    "seaborn",
                    "networkx",
                    "scipy",
                    "pytest",
                    "pandas"]


setup(name = name,
      url = url,
      author = author,
      author_email = author_email,
      version = version,
      description = description,
      ext_modules = ext_modules,
      package_data = package_data,
      package_dir = package_dir,
      packages = packages,
      entry_points = entry_points,
      install_requires = install_requires,
)

