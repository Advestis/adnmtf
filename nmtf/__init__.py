"""
Non-negative Matrix and Tensor Factorization
Author: Paul Fogel
License: MIT
Release date: Jan 6 '20
Version: 11.0.0
https://github.com/paulfogel/NMTF
"""

from .nmtf import NMF, NTF

try:
    from ._version import __version__
except ImportError:
    pass
