from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy

setup(
    name='graphormer_algos',
    ext_modules=cythonize([
        Extension(
            "graphormer.data.algos",
            ["graphormer/data/algos.pyx"],
            include_dirs=[numpy.get_include()],
        )
    ]),
    zip_safe=False,
)
