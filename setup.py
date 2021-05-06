from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext as _build_ext

# this is to avoid import numpy as first thing in the install script, which
# fails if numpy isn't installed already. Same thing that root_numpy does
# and similar to other sources
# https://github.com/scikit-hep/root_numpy

class build_ext(_build_ext):
    def finalize_options(self):
        _build_ext.finalize_options(self)

        # prevent numpy from thinking it is still in its setup process:
        try:
            del builtins.__NUMPY_SETUP__
        except AttributeError:
            pass

        import numpy
        self.include_dirs.append(numpy.get_include())

cmdclass={'build_ext': _build_ext},

libinteract = Extension('libinteract.innerloops', ['libinteract/innerloops.pyx', 'libinteract/clibinteract.c'])

setup(name = 'pyinteraph',
      url='https://www.github.com/ELELAB/pyinteraph',
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
                         "filter_graph = pyinteraph.filter_graph:main",
                         "parse_masses = pyinteraph.parse_masses:main",
                         "path_analysis = pyinteraph.path_analysis:main",
                         "centrality_analysis = pyinteraph.centrality_analysis:main"]
                   },
      install_requires=["cython",
                        "biopython",
                        "MDAnalysis==1.0.0",
                        "numpy",
                        "matplotlib",
                        "seaborn",
                        "networkx",
                        "scipy",
                        "pytest",
                        "pandas"],
      setup_requires=["numpy",
                      "cython"],
      cmd_class={'build_ext' : build_ext}
      )
