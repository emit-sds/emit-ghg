"""
EMIT GHG - Greenhouse Gas Detection from EMIT Hyperspectral Data

Point-source methane and CO2 detection using matched filter algorithms.
"""

__version__ = "0.0.2"
__author__ = "EMIT Team"

# Import key classes/functions for convenience
from .files import Filenames

__all__ = ["Filenames", "__version__"]
