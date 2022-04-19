"""
Non-negative Matrix and Tensor Factorization\n
Author: Paul Fogel\n
License: MIT\n
https://github.com/Advestis/adnmtf
"""

from .nmtf import NMF, NTF

from . import _version
__version__ = _version.get_versions()['version']
