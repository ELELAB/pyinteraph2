# -*- coding: utf-8 -*-

from setuptools import setup, Extension

libinteract = Extension('libinteract.innerloops', ['libinteract/innerloops.pyx', 'libinteract/clibinteract.c'])

setup(name = 'pyinteraph',
      url='http://linux.btbs.unimib.it/pyinteraph/',
      author="Matteo Tiberti, Gaetano Invernizzi, Yuval Inbar, Matteo Lambrughi, Gideon Schreiber, Elena Papaleo",
      author_email="matteo.tiberti@gmail.com",
      version = '1.1',
      description = 'Compute interatomic interactions in protein ensembles',
      ext_modules = [libinteract],
      package_data = {'pyinteraph' : ['charged_groups.ini',  'hydrogen_bonds.ini', 'kbp_atomlist', 'charged_groups.ini', 'hydrogen_bonds.ini', 'ff.S050.bin64', 'ff_masses/*']},
      package_dir={'libinteract':'libinteract', 'pyinteraph':'pyinteraph'},
      packages=['libinteract', 'pyinteraph'],
      entry_points={ "console_scripts" : [
                         "pyinteraph = pyinteraph.pyinteraph:main",
                         "graph_analysis = pyinteraph.graph_analysis:main",
                         "filter_graph = pyinteraph.filter_graph:main"] 
                   },
      install_requires=["biopython",
                        "MDAnalysis==1.0.0",
                        "numpy",
                        "matplotlib",
                        "networkx",
                        "scipy",
                        "pytest"],
      setup_requires=["numpy"]
      )
