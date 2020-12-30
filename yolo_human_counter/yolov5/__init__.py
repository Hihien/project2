import os
import sys

sys.path.append(os.path.dirname(__file__))
del os, sys

from .detector import Detector

__all__ = ['Detector']
