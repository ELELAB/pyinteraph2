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
      data_files=[('pyinteraph',         ['charged_groups.ini',  'hydrogen_bonds.ini', 'kbp_atomlist', 'charged_groups.ini', 'hydrogen_bonds.ini', 'ff.S050.bin64']),
                  ('pyinteraph/ff_masses',['ff_masses/amber03', 'ff_masses/amber94', 'ff_masses/amber96', 'ff_masses/amber99', 'ff_masses/amber99sb', 'ff_masses/amber99sb-ildn', 'ff_masses/amberGS', 'ff_masses/charmm27', 'ff_masses/encads', 'ff_masses/encadv', 'ff_masses/gromos43a1', 'ff_masses/gromos43a2', 'ff_masses/gromos45a3', 'ff_masses/gromos53a5', 'ff_masses/gromos53a6', 'ff_masses/oplsaa']),              ],
      package_dir={'libinteract':'libinteract', 'pyinteraph':'pyinteraph'},
      packages=['libinteract', 'pyinteraph'],
      entry_points={ "console_scripts" : [
                         "pyinteraph = pyinteraph.pyinteraph:main",
                         "graph_analysis = pyinteraph.graph_analysis:main",
                         "filter_graph = pyinteraph.filter_graph:main"] 
                   },
      include_dirs=[numpy.get_include()]
      
      )
