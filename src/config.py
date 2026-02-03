"""
Configuration module for InSAR Soil Moisture processing.

This module provides configuration classes and utilities for managing
processing parameters.
"""

import os
import yaml
from dataclasses import dataclass, field
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


@dataclass
class Config:
    """
    Configuration class for InSAR Soil Moisture processing.

    Attributes:
        WINDOW_RANGE: Multi-looking window size in range direction (pixels)
        WINDOW_AZIMUTH: Multi-looking window size in azimuth direction (pixels)
        TRIPLET_COHERENCE_THRESHOLD: Minimum product of 3 coherences for triplet selection
        AGGREGATION_COHERENCE_THRESHOLD: Minimum coherence for output pixels
        OUTPUT_SPACING: Output grid spacing in meters
        TEMPORAL_FILTER_T: Characteristic time for exponential temporal filter (days)
        SPATIAL_FILTER_SIGMA: Standard deviation for Gaussian spatial filter (pixels)
        N_PARALLEL: Number of parallel workers (0 = auto)
        CHUNK_SIZE: Processing chunk size for memory management
        OUTPUT_FORMAT: Output raster format
        COMPRESS: Compression algorithm for output
        NODATA_VALUE: NoData value for output rasters

    Example:
        >>> config = Config()
        >>> config.WINDOW_RANGE
        79
        >>> config = Config(WINDOW_RANGE=100, TEMPORAL_FILTER_T=5)
        >>> config.TEMPORAL_FILTER_T
        5
    """

    # Multi-looking window (79 range x 18 azimuth ≈ 240m ground for Sentinel-1)
    WINDOW_RANGE: int = 79
    WINDOW_AZIMUTH: int = 18

    # Coherence thresholds
    TRIPLET_COHERENCE_THRESHOLD: float = 0.027  # 0.3^3
    AGGREGATION_COHERENCE_THRESHOLD: float = 0.1

    # Output grid spacing (meters)
    OUTPUT_SPACING: int = 1000  # 1 km

    # Temporal filter characteristic time (days)
    TEMPORAL_FILTER_T: float = 3.0
    TEMPORAL_FILTER_ENABLED: bool = True

    # Spatial filter sigma (pixels)
    SPATIAL_FILTER_SIGMA: float = 3.0
    SPATIAL_FILTER_ENABLED: bool = True

    # Parallel processing
    N_PARALLEL: int = 0  # 0 = auto-detect
    CHUNK_SIZE: int = 1000  # Pixels per chunk

    # Output settings
    OUTPUT_FORMAT: str = "GTiff"
    COMPRESS: str = "LZW"
    NODATA_VALUE: float = -9999.0

    # Quality control
    MIN_VALID_PIXELS: int = 100  # Minimum valid pixels per window
    MAX_CLOSURE_PHASE: float = 3.14159  # Maximum valid closure phase (pi)

    # Verbosity
    LOG_LEVEL: str = "INFO"
    PROGRESS_BAR: bool = True

    def __post_init__(self):
        """Validate configuration after initialization."""
        self._validate()

    def _validate(self):
        """Validate configuration parameters."""
        if self.WINDOW_RANGE < 1:
            raise ValueError(f"WINDOW_RANGE must be positive, got {self.WINDOW_RANGE}")

        if self.WINDOW_AZIMUTH < 1:
            raise ValueError(f"WINDOW_AZIMUTH must be positive, got {self.WINDOW_AZIMUTH}")

        if not 0 < self.TRIPLET_COHERENCE_THRESHOLD < 1:
            raise ValueError(
                f"TRIPLET_COHERENCE_THRESHOLD must be in (0, 1), "
                f"got {self.TRIPLET_COHERENCE_THRESHOLD}"
            )

        if not 0 < self.AGGREGATION_COHERENCE_THRESHOLD < 1:
            raise ValueError(
                f"AGGREGATION_COHERENCE_THRESHOLD must be in (0, 1), "
                f"got {self.AGGREGATION_COHERENCE_THRESHOLD}"
            )

        if self.OUTPUT_SPACING < 1:
            raise ValueError(f"OUTPUT_SPACING must be positive, got {self.OUTPUT_SPACING}")

        if self.TEMPORAL_FILTER_T <= 0:
            raise ValueError(
                f"TEMPORAL_FILTER_T must be positive, got {self.TEMPORAL_FILTER_T}"
            )

        if self.SPATIAL_FILTER_SIGMA <= 0:
            raise ValueError(
                f"SPATIAL_FILTER_SIGMA must be positive, got {self.SPATIAL_FILTER_SIGMA}"
            )

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "Config":
        """
        Load configuration from YAML file.

        Args:
            yaml_path: Path to YAML configuration file

        Returns:
            Config instance with values from YAML file

        Example:
            >>> config = Config.from_yaml('config.yaml')
        """
        if not os.path.exists(yaml_path):
            raise FileNotFoundError(f"Configuration file not found: {yaml_path}")

        with open(yaml_path, 'r') as f:
            yaml_config = yaml.safe_load(f)

        return cls.from_dict(yaml_config)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "Config":
        """
        Create configuration from dictionary.

        Handles nested dictionaries (e.g., from YAML files).

        Args:
            config_dict: Dictionary with configuration values

        Returns:
            Config instance
        """
        flat_config = {}

        # Flatten nested config
        if 'processing' in config_dict:
            proc = config_dict['processing']
            if 'window_range' in proc:
                flat_config['WINDOW_RANGE'] = proc['window_range']
            if 'window_azimuth' in proc:
                flat_config['WINDOW_AZIMUTH'] = proc['window_azimuth']

        if 'thresholds' in config_dict:
            thresh = config_dict['thresholds']
            if 'triplet_coherence' in thresh:
                flat_config['TRIPLET_COHERENCE_THRESHOLD'] = thresh['triplet_coherence']
            if 'aggregation_coherence' in thresh:
                flat_config['AGGREGATION_COHERENCE_THRESHOLD'] = thresh['aggregation_coherence']

        if 'output' in config_dict:
            out = config_dict['output']
            if 'spacing' in out:
                flat_config['OUTPUT_SPACING'] = out['spacing']
            if 'format' in out:
                flat_config['OUTPUT_FORMAT'] = out['format']
            if 'compress' in out:
                flat_config['COMPRESS'] = out['compress']

        if 'filtering' in config_dict:
            filt = config_dict['filtering']
            if 'temporal' in filt:
                temp = filt['temporal']
                if 'enabled' in temp:
                    flat_config['TEMPORAL_FILTER_ENABLED'] = temp['enabled']
                if 'characteristic_time' in temp:
                    flat_config['TEMPORAL_FILTER_T'] = temp['characteristic_time']
            if 'spatial' in filt:
                spat = filt['spatial']
                if 'enabled' in spat:
                    flat_config['SPATIAL_FILTER_ENABLED'] = spat['enabled']
                if 'sigma' in spat:
                    flat_config['SPATIAL_FILTER_SIGMA'] = spat['sigma']

        # Also handle flat dictionary input
        for key in ['WINDOW_RANGE', 'WINDOW_AZIMUTH', 'TRIPLET_COHERENCE_THRESHOLD',
                    'AGGREGATION_COHERENCE_THRESHOLD', 'OUTPUT_SPACING',
                    'TEMPORAL_FILTER_T', 'SPATIAL_FILTER_SIGMA', 'N_PARALLEL',
                    'CHUNK_SIZE', 'OUTPUT_FORMAT', 'COMPRESS', 'NODATA_VALUE',
                    'TEMPORAL_FILTER_ENABLED', 'SPATIAL_FILTER_ENABLED',
                    'LOG_LEVEL', 'PROGRESS_BAR']:
            if key in config_dict:
                flat_config[key] = config_dict[key]

        return cls(**flat_config)

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to dictionary.

        Returns:
            Dictionary representation of configuration
        """
        return {
            'processing': {
                'window_range': self.WINDOW_RANGE,
                'window_azimuth': self.WINDOW_AZIMUTH,
            },
            'thresholds': {
                'triplet_coherence': self.TRIPLET_COHERENCE_THRESHOLD,
                'aggregation_coherence': self.AGGREGATION_COHERENCE_THRESHOLD,
            },
            'output': {
                'spacing': self.OUTPUT_SPACING,
                'format': self.OUTPUT_FORMAT,
                'compress': self.COMPRESS,
                'nodata': self.NODATA_VALUE,
            },
            'filtering': {
                'temporal': {
                    'enabled': self.TEMPORAL_FILTER_ENABLED,
                    'characteristic_time': self.TEMPORAL_FILTER_T,
                },
                'spatial': {
                    'enabled': self.SPATIAL_FILTER_ENABLED,
                    'sigma': self.SPATIAL_FILTER_SIGMA,
                },
            },
            'parallel': {
                'n_workers': self.N_PARALLEL,
                'chunk_size': self.CHUNK_SIZE,
            },
        }

    def to_yaml(self, yaml_path: str) -> None:
        """
        Save configuration to YAML file.

        Args:
            yaml_path: Output path for YAML file
        """
        with open(yaml_path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)

        logger.info(f"Configuration saved to {yaml_path}")

    def get_window_area(self) -> int:
        """Get total number of pixels in multi-look window."""
        return self.WINDOW_RANGE * self.WINDOW_AZIMUTH

    def get_ground_resolution(self, pixel_spacing: float = 2.3) -> float:
        """
        Calculate approximate ground resolution of multi-look window.

        Args:
            pixel_spacing: SLC pixel spacing in meters (default for Sentinel-1 IW)

        Returns:
            Approximate ground resolution in meters
        """
        # Assuming square resolution cell
        range_res = self.WINDOW_RANGE * pixel_spacing
        azimuth_res = self.WINDOW_AZIMUTH * pixel_spacing * 4  # Sentinel-1 azimuth oversampling
        return (range_res + azimuth_res) / 2

    def __str__(self) -> str:
        """String representation of configuration."""
        lines = ["InSAR Soil Moisture Configuration:"]
        lines.append(f"  Window: {self.WINDOW_RANGE} x {self.WINDOW_AZIMUTH} pixels")
        lines.append(f"  Coherence thresholds: triplet={self.TRIPLET_COHERENCE_THRESHOLD:.3f}, "
                    f"aggregation={self.AGGREGATION_COHERENCE_THRESHOLD:.2f}")
        lines.append(f"  Output spacing: {self.OUTPUT_SPACING} m")
        lines.append(f"  Temporal filter: {'enabled' if self.TEMPORAL_FILTER_ENABLED else 'disabled'} "
                    f"(T={self.TEMPORAL_FILTER_T} days)")
        lines.append(f"  Spatial filter: {'enabled' if self.SPATIAL_FILTER_ENABLED else 'disabled'} "
                    f"(sigma={self.SPATIAL_FILTER_SIGMA} pixels)")
        return "\n".join(lines)


def get_default_config() -> Config:
    """
    Get default configuration.

    Returns:
        Config instance with default values
    """
    return Config()


def create_example_config(output_path: str) -> None:
    """
    Create example configuration file.

    Args:
        output_path: Path for output YAML file
    """
    example = """# InSAR Soil Moisture Configuration
# ===================================

# Processing parameters
processing:
  window_range: 79          # Multi-look window size (range direction, pixels)
  window_azimuth: 18        # Multi-look window size (azimuth direction, pixels)
  # These values correspond to ~240m ground resolution for Sentinel-1 IW mode

# Coherence thresholds
thresholds:
  triplet_coherence: 0.027  # Product of 3 coherences (default: 0.3^3)
  aggregation_coherence: 0.1  # Minimum coherence for output pixels

# Output settings
output:
  spacing: 1000             # Output grid spacing in meters (1 km)
  format: GTiff             # Output format (GTiff, NetCDF, etc.)
  compress: LZW             # Compression algorithm

# Filtering options
filtering:
  temporal:
    enabled: true
    characteristic_time: 3  # Exponential filter time constant (days)
  spatial:
    enabled: true
    sigma: 3                # Gaussian filter standard deviation (pixels)

# Parallel processing
parallel:
  n_workers: 0              # Number of workers (0 = auto-detect)
  chunk_size: 1000          # Pixels per chunk for memory management
"""

    with open(output_path, 'w') as f:
        f.write(example)

    logger.info(f"Example configuration created at {output_path}")
