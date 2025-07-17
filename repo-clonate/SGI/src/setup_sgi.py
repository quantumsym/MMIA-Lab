from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy


ext_modules = [
    Extension(
        "sgi",
        ["./src/sgi.pyx"],
        include_dirs=[numpy.get_include()],
        extra_compile_args=['-fopenmp'],
        extra_link_args=['-fopenmp'],
    )
]

setup(name="sgi",
      ext_modules=cythonize(ext_modules))

