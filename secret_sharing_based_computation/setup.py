
#添加 include_dirs=[np.get_include()]


from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy as np


ext = [
        Extension(name="_basic_building_blocks", sources=["_basic_building_blocks.pyx"]),
        Extension(name="_sfc", sources=["_sfc.pyx"]) 
      ]



for e in ext:
    e.cython_directives = {'language_level': "3"} #all are Python-3


setup(
    cmdclass = {'build_ext': build_ext},
    ext_modules = ext,
    include_dirs=[np.get_include()]
)