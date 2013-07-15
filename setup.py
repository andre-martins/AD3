from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
from distutils.extension import Extension
from distutils.command.build_clib import build_clib
from distutils.errors import DistutilsSetupError
from distutils import log


class build_libad3(build_clib):
    def build_libraries(self, libraries):
        for (lib_name, build_info) in libraries:
            sources = build_info.get('sources')
            if sources is None or not isinstance(sources, (list, tuple)):
                raise DistutilsSetupError(
                    "in 'libraries' option (library '%s'), " +
                    "'sources' must be present and must be " +
                    "a list of source filenames" % lib_name)
            sources = list(sources)

            log.info("building '%s' library", lib_name)

            # First, compile the source code to object files in the library
            # directory. (This should probably change to putting object
            # files in a temporary build directory.)
            macros = build_info.get('macros')
            include_dirs = build_info.get('include_dirs')
            extra_compile_args = build_info.get('extra_compile_args')
            objects = self.compiler.compile(sources,
                                            output_dir=self.build_temp,
                                            macros=macros,
                                            include_dirs=include_dirs,
                                            extra_preargs=extra_compile_args,
                                            debug=self.debug,
                                            )

            # Now "link" the object files together into a static library.
            # (On Unix at least, this isn't really linking -- it just
            # builds an archive. Whatever.)
            self.compiler.create_static_lib(objects, lib_name,
                                            output_dir=self.build_clib,
                                            debug=self.debug)
libad3 = ('ad3', {
    'sources': ['ad3/FactorGraph.cpp',
                'ad3/GenericFactor.cpp',
                'ad3/Factor.cpp',
                'ad3/Utils.cpp'],
    'include_dirs': ['.',
                     './ad3',
                     './Eigen'
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


setup(name='ad3',
      package_dir = {'ad3': 'python/ad3'},
      packages = ['ad3'],
      libraries=[libad3],
      cmdclass={'build_clib': build_libad3, 'build_ext' : build_ext},
      ext_modules=[Extension("ad3.factor_graph", 
                             ["python/factor_graph.pyx"], 
                             include_dirs = ["ad3"],
                             language="c++",
                             )])

