
#添加 include_dirs=[np.get_include()]


from setuptools import setup, Extension
# from distutils.extension import Extension
# from Cython.Distutils import build_ext
from Cython.Build import cythonize
import numpy as np


ext = [
        Extension(name="_base_test",
        sources=["_base_test.pyx"]) 
      ]



for e in ext:
    e.cython_directives = {'language_level': "3"} #all are Python-3


setup(
    # cmdclass = {'build_ext': build_ext},
    ext_modules = cythonize(ext[0]),
    include_dirs=[np.get_include()]
)


setup(
    # cmdclass = {'build_ext': build_ext},
    ext_modules = cythonize(ext),
    include_dirs=[np.get_include()]
)