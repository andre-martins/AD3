import sys
from setuptools import setup
from setuptools.extension import Extension
from setuptools.command.build_ext import build_ext
from setuptools.command.build_clib import build_clib
from setuptools.command.bdist_egg import bdist_egg


AD3_FLAGS_UNIX = [
    '-O3',
    '-Wall',
    '-Wno-sign-compare',
    '-Wno-overloaded-virtual',
    '-c',
    '-fmessage-length=0',
    '-fPIC',
    '-ffast-math',
    '-march=native'
]


AD3_FLAGS_MSVC = [
    '/O2',
    '/fp:fast',
    '/favor:INTEL64',
    '/wd4267'  # suppress sign-compare--like warning
]


AD3_CFLAGS =  {
    'cygwin' : AD3_FLAGS_UNIX,
    'mingw32' : AD3_FLAGS_UNIX,
    'unix' : AD3_FLAGS_UNIX,
    'msvc' : AD3_FLAGS_MSVC
}


# support compiler-specific cflags in extensions and libs
class our_build_ext(build_ext):
    def build_extensions(self):

        # bug in distutils: flag not valid for c++
        flag = '-Wstrict-prototypes'
        if (hasattr(self.compiler, 'compiler_so')
                and flag in self.compiler.compiler_so):
            self.compiler.compiler_so.remove(flag)

        compiler_type = self.compiler.compiler_type
        compile_args = AD3_CFLAGS.get(compiler_type, [])

        for e in self.extensions:
            e.extra_compile_args.extend(compile_args)

        build_ext.build_extensions(self)


class our_build_clib(build_clib):
    def build_libraries(self, libraries):

        # bug in distutils: flag not valid for c++
        flag = '-Wstrict-prototypes'
        if (hasattr(self.compiler, 'compiler_so')
                and flag in self.compiler.compiler_so):
            self.compiler.compiler_so.remove(flag)

        compiler_type = self.compiler.compiler_type
        compile_args = AD3_CFLAGS.get(compiler_type, [])

        for (lib_name, build_info) in libraries:
            build_info['cflags'] = compile_args

        build_clib.build_libraries(self, libraries)


# this is a backport of a workaround for a problem in distutils.
# install_lib doesn't call build_clib
class our_bdist_egg(bdist_egg):
    def run(self):
        self.call_command('build_clib')
        bdist_egg.run(self)


cmdclass = {
    'build_ext': our_build_ext,
    'build_clib': our_build_clib,
    'bdist_egg': our_bdist_egg}


WHEELHOUSE_UPLOADER_COMMANDS = set(['fetch_artifacts', 'upload_all'])
if WHEELHOUSE_UPLOADER_COMMANDS.intersection(sys.argv):
    import wheelhouse_uploader.cmd
    cmdclass.update(vars(wheelhouse_uploader.cmd))


libad3 = ('ad3', {
    'language': "c++",
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
})


setup(name='ad3',
      version="2.3.dev0",
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
                    language="c++")
          ])
