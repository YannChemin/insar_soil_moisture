"""
InSAR Soil Moisture Retrieval Package

A Python implementation of the InSAR-based surface soil moisture retrieval
algorithm based on SAR interferometry and closure phases.

Reference:
    De Zan, F., Filippucci, P., & Brocca, L. (2026). Validation of high-resolution
    surface soil moisture time series retrieved by means of SAR interferometry.
    Remote Sensing of Environment, 335, 115266.
"""

__version__ = "1.0.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

from .config import Config
from .processor import InSARSoilMoistureProcessor
from .postprocessing import PostProcessor
from .io_utils import read_slc_stack, write_geotiff

__all__ = [
    "Config",
    "InSARSoilMoistureProcessor",
    "PostProcessor",
    "read_slc_stack",
    "write_geotiff",
]
