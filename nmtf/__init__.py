"""
Non-negative Matrix and Tensor Factorization\n
Author: Paul Fogel\n
License: MIT\n
https://github.com/Advestis/nmtf_private
"""

from .nmtf import NMF, NTF

try:
    from ._version import __version__
except ImportError:
    pass
