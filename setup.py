import sys
from setuptools import setup
from setuptools.command.bdist_egg import bdist_egg
from setuptools.extension import Extension


AD3_COMPILE_ARGS = [
    '-fPIC',
    '-O3',
    '-c',
    '-fmessage-length=0'
]

libad3 = ('ad3', {
    'sources': ['ad3/FactorGraph.cpp',
                'ad3/GenericFactor.cpp',
                'ad3/Factor.cpp',
                'ad3/Utils.cpp',
                'examples/cpp/parsing/FactorTree.cpp'
                ],
    'include_dirs': ['.',
                     './ad3',
                     './Eigen',
                     './examples/cpp/parsing'
                     ],
    'extra_compile_args': AD3_COMPILE_ARGS
})

# this is a backport of a workaround for a problem in distutils.
# install_lib doesn't call build_clib


class bdist_egg_fix(bdist_egg):
    def run(self):
        self.call_command('build_clib')
        bdist_egg.run(self)


WHEELHOUSE_UPLOADER_COMMANDS = set(['fetch_artifacts', 'upload_all'])

cmdclass = {'bdist_egg': bdist_egg_fix}

if WHEELHOUSE_UPLOADER_COMMANDS.intersection(sys.argv):

    import wheelhouse_uploader.cmd
    cmdclass.update(vars(wheelhouse_uploader.cmd))


setup(name='ad3',
      version="2.2rc1",
      author="Andre Martins",
      description='Alternating Directions Dual Decomposition',
      url="http://www.ark.cs.cmu.edu/AD3",
      author_email="afm@cs.cmu.edu",
      package_dir={
          'ad3': 'python/ad3',
          'ad3.tests': 'python/ad3/tests'
      },
      packages=['ad3', 'ad3.tests'],
      libraries=[libad3],
      cmdclass=cmdclass,
      include_package_data=True,
      ext_modules=[
          Extension("ad3.factor_graph",
                    ["python/ad3/factor_graph.cpp"],
                    include_dirs=[".", "ad3"],
                    language="c++"),
          Extension("ad3.base",
                    ["python/ad3/base.cpp"],
                    include_dirs=[".", "ad3"],
                    language="c++"),
          Extension("ad3.extensions",
                    ["python/ad3/extensions.cpp"],
                    include_dirs=[".", "ad3"],
                    language="c++",
                    extra_compile_args=AD3_COMPILE_ARGS),
          ])
