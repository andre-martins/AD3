from setuptools import setup
from setuptools.command.bdist_egg import bdist_egg
from setuptools.extension import Extension

libad3 = ('ad3', {
    'sources': ['ad3/FactorGraph.cpp',
                'ad3/GenericFactor.cpp',
                'ad3/Factor.cpp',
                'ad3/Utils.cpp',
                'examples/cpp/parsing/FactorTree.cpp'],
    'include_dirs': ['.',
                     './ad3',
                     './Eigen',
                     './examples/cpp/parsing'
                     ],
    'extra_compile_args': [
        '-Wno-sign-compare',
        '-Wall',
        '-fPIC',
        '-O3',
        '-c',
        '-fmessage-length=0'
    ],
})

# this is a backport of a workaround for a problem in distutils.
# install_lib doesn't call build_clib


class bdist_egg_fix(bdist_egg):
  def run(self):
      self.call_command('build_clib')
      bdist_egg.run(self)


setup(name='ad3',
      version="2.1.dev0",
      author="Andre Martins",
      description='Alternating Directions Dual Decomposition',
      url="http://www.ark.cs.cmu.edu/AD3",
      author_email="afm@cs.cmu.edu",
      package_dir={'ad3': 'python', 'ad3/tests': 'python/tests'},
      packages=['ad3', 'ad3/tests'],
      libraries=[libad3],
      cmdclass={'bdist_egg': bdist_egg_fix},
      ext_modules=[Extension("ad3.factor_graph",
                             ["python/factor_graph.cpp"],
                             include_dirs=[".", "ad3"],
                             language="c++",
                             )])
