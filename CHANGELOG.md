# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2024-02-03

### Added

- Initial release of InSAR Soil Moisture retrieval package
- Core InSAR processing based on De Zan et al. (2026) algorithm
  - Closure phase-based triplet selection
  - Soil moisture index inversion
  - Synthetic reference generation
  - Phase history extraction
- Post-processing capabilities
  - Sign correction using scene-average correlation
  - Coherence-based quality masking
  - Exponential temporal filtering (SWI-like)
  - Gaussian spatial filtering
  - Grid aggregation
- Input/Output utilities
  - SLC stack reading (GeoTIFF format)
  - Multi-band GeoTIFF output
  - NetCDF output support
  - VRT stack creation
- Command-line interface
  - `insar-sm process` - Main processing command
  - `insar-sm init-config` - Create example configuration
  - `insar-sm info` - Display raster information
  - `insar-sm plot` - Quick visualization
- Configuration system
  - YAML configuration file support
  - Comprehensive parameter validation
- Documentation
  - README with algorithm overview
  - Installation guide
  - API documentation
  - Example scripts

### Dependencies

- numpy >= 1.21.0
- scipy >= 1.7.0
- GDAL >= 3.0.0
- pyyaml >= 6.0
- click >= 8.0.0

## [Unreleased]

### Planned

- Tiled processing for large datasets
- Parallel processing support
- Integration with ISCE2 for preprocessing
- L-band SAR support (NISAR, ROSE-L)
- Backscatter-phase fusion capability
- Web visualization interface
