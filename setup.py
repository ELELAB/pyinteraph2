# -*- coding: utf-8 -*-

from setuptools import setup, Extension
import numpy

c_module = Extension('libinteract.innerloops', ['src/innerloops.c', 'src/clibinteract.c'])

setup(name = 'pyinteraph',
      url='http://linux.btbs.unimib.it/pyinteraph/',
      author="Matteo Tiberti, Gaetano Invernizzi, Yuval Inbar, Matteo Lambrughi, Gideon Schreiber, Elena Papaleo",
      author_email="matteo.tiberti@gmail.com",
      version = '1.1',
      description = 'Compute interatomic interactions in protein ensembles',
      ext_modules = [c_module],
      package_data = {'pyinteraph' : ['charged_groups.ini',  'hydrogen_bonds.ini', 'kbp_atomlist', 'charged_groups.ini', 'hydrogen_bonds.ini', 'ff.S050.bin64', 'ff_masses/*']},
      package_dir={'libinteract':'libinteract', 'pyinteraph':'pyinteraph'},
      packages=['libinteract', 'pyinteraph'],
      entry_points={ "console_scripts" : [
                         "pyinteraph = pyinteraph.pyinteraph:main",
                         "graph_analysis = pyinteraph.graph_analysis:main",
                         "filter_graph = pyinteraph.filter_graph:main"] 
                   },
      include_dirs=[numpy.get_include()]
      )
