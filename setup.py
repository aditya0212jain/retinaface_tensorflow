from distutils.core import setup
from Cython.Build import cythonize

setup(
    name='iou',
    ext_modules = cythonize("iou.pyx")
)