# InSAR Soil Moisture Retrieval

A Python implementation of the InSAR-based surface soil moisture retrieval algorithm based on SAR interferometry and closure phases, as described in:

> De Zan, F., Filippucci, P., & Brocca, L. (2026). *Validation of high-resolution surface soil moisture time series retrieved by means of SAR interferometry*. Remote Sensing of Environment, 335, 115266.

## Overview

This tool retrieves high-resolution (1 km) soil moisture time series from Sentinel-1 SAR imagery using interferometric closure phases. Unlike traditional backscatter-based methods, this approach exploits the phase information in SAR images, which is highly sensitive to soil water content.

### Key Features

- Closure phase-based soil moisture retrieval
- Efficient processing of long SAR time series
- Configurable spatial and temporal filtering
- Quality metrics based on interferometric coherence
- GeoTIFF output compatible with standard GIS tools
- Support for batch processing of large areas

### Algorithm Workflow

```
┌─────────────────┐
│  Co-registered  │
│   SLC Stack     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Triplet Select  │  Find consecutive acquisitions with
│ (max closure φ) │  largest closure phase
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Moisture Order  │  Assign nominal levels using
│   Inversion     │  coherence + closure phase sign
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Reference     │  Generate synthetic reference
│   Generation    │  images z_a and z_b
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Phase History   │  Compute interferograms with
│   Extraction    │  both references
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Soil Moisture   │  m(t) ∝ φ_a(t) - φ_b(t)
│    Index        │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Post-processing │  Sign correction, filtering,
│                 │  aggregation to 1km grid
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Output GeoTIFF │
│ (SM time series)│
└─────────────────┘
```

## Installation

See [INSTALL.md](INSTALL.md) for detailed installation instructions.

### Quick Start

```bash
# Clone repository
git clone https://github.com/yannchemin/insar_soil_moisture.git
cd insar_soil_moisture

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Install package
pip install -e .
```

## Usage

### Command Line Interface

```bash
# Basic usage with co-registered SLC stack
insar-sm --slc-dir /path/to/coregistered_slcs --output-dir /path/to/output

# With custom configuration
insar-sm --slc-dir /path/to/slcs --output-dir /path/to/output \
    --config config.yaml \
    --no-temporal-filter \
    --parallel 8

# Process specific date range
insar-sm --slc-dir /path/to/slcs --output-dir /path/to/output \
    --start-date 2020-01-01 --end-date 2020-12-31
```

### Python API

```python
from insar_soil_moisture import InSARSoilMoistureProcessor, Config

# Initialize with custom configuration
config = Config(
    window_range=79,
    window_azimuth=18,
    output_spacing=1000,
    temporal_filter_t=3,
    spatial_filter_sigma=3
)

# Process SLC stack
processor = InSARSoilMoistureProcessor(config)
result = processor.process(
    slc_paths=['slc_001.tif', 'slc_002.tif', ...],
    output_dir='./output'
)

# Access results
print(f"Output saved to: {result.output_path}")
print(f"Mean coherence: {result.mean_coherence:.2f}")
print(f"Valid pixels: {result.valid_fraction*100:.1f}%")
```

### Configuration File

Create a `config.yaml` file for reproducible processing:

```yaml
# Processing parameters
processing:
  window_range: 79          # Multi-look window size (range)
  window_azimuth: 18        # Multi-look window size (azimuth)

# Coherence thresholds
thresholds:
  triplet_coherence: 0.027  # Product of 3 coherences (0.3^3)
  aggregation_coherence: 0.1

# Output settings
output:
  spacing: 1000             # Output grid spacing (meters)
  format: GTiff             # Output format
  compress: LZW             # Compression

# Filtering
filtering:
  temporal:
    enabled: true
    characteristic_time: 3  # Days
  spatial:
    enabled: true
    sigma: 3                # Pixels
```

## Input Requirements

### Pre-processing with ISCE2

The algorithm requires **co-registered SLC data**. Use ISCE2 for pre-processing:

```bash
# 1. Download Sentinel-1 SLC products from ASF or Copernicus
# 2. Prepare stack with ISCE2
stackSentinel.py \
    -s ./SLCs \
    -d ./DEM/dem.wgs84 \
    -a ./AUX \
    -o ./stack \
    -c ./coregistered

# 3. Convert to GeoTIFF (if needed)
python scripts/slc_to_geotiff.py ./coregistered ./geotiff_stack
```

### Pre-processing with SNAP

Alternatively, use ESA SNAP:

```bash
# Using SNAP GPT
gpt CoregistrationGraph.xml \
    -Pmaster=S1A_20200101.zip \
    -Pslave=S1A_20200113.zip \
    -Poutput=coregistered.dim
```

### Input Data Format

| Parameter | Requirement |
|-----------|-------------|
| Format | GeoTIFF (complex or I/Q bands) |
| Projection | Any projected CRS (UTM recommended) |
| Co-registration | Sub-pixel accuracy required |
| Stack size | Minimum 3 images, recommended 20+ |
| Temporal baseline | 6-12 days between acquisitions |

## Output Products

| File | Description |
|------|-------------|
| `insar_soil_moisture.tif` | Multi-band GeoTIFF with SM time series |
| `coherence_map.tif` | Mean temporal coherence |
| `quality_flags.tif` | Quality indicators |
| `metadata.json` | Processing parameters and dates |

### Interpreting Results

The output soil moisture index is **relative** (not volumetric). To convert to volumetric soil moisture:

```python
from insar_soil_moisture.calibration import calibrate_with_insitu

# Calibrate using in-situ measurements
sm_volumetric = calibrate_with_insitu(
    sm_index='insar_soil_moisture.tif',
    insitu_data='station_data.csv',
    method='linear'  # or 'cdf_matching'
)
```

## Performance Considerations

### Memory Usage

For large areas, use tiled processing:

```python
processor.process(
    slc_paths=slc_list,
    output_dir='./output',
    tile_size=(1000, 1000),  # Process in tiles
    overlap=100              # Tile overlap in pixels
)
```

### Parallel Processing

Enable multi-core processing:

```bash
insar-sm --slc-dir ./slcs --output-dir ./output --parallel 8
```

### Processing Time Estimates

| Stack Size | Area | Cores | Time |
|------------|------|-------|------|
| 50 images | 100x100 km | 1 | ~4 hours |
| 50 images | 100x100 km | 8 | ~45 min |
| 100 images | 200x200 km | 8 | ~3 hours |

## Validation

The algorithm has been validated against:
- In-situ soil moisture stations (R = 0.32-0.78)
- ERA5-Land modeled soil moisture (R = 0.53-0.87)
- SMAP 1km soil moisture (R = 0.57-0.84)

Best performance is achieved in areas with:
- High interferometric coherence (> 0.3)
- Sparse vegetation cover
- Flat to moderate topography
- Low urbanization

## Limitations

- Requires co-registered SLC data (pre-processing step)
- Performance degrades in vegetated or urban areas
- Returns relative (not absolute) soil moisture
- Limited performance over snow-covered surfaces

## References

1. De Zan, F., Filippucci, P., & Brocca, L. (2026). Validation of high-resolution surface soil moisture time series retrieved by means of SAR interferometry. *Remote Sensing of Environment*, 335, 115266.

2. De Zan, F., & Gomba, G. (2018). Vegetation and soil moisture inversion from SAR closure phases. *Remote Sensing of Environment*, 217, 562-572.

3. De Zan, F., Parizzi, A., Prats-Iraola, P., & López-Dekker, P. (2014). A SAR interferometric model for soil moisture. *IEEE TGRS*, 52(1), 418-425.

## License

This project is licensed under the Unlicense - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please read [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## Acknowledgments

This implementation is based on research supported by:
- ESA 4DMED-DEMETRAS project
- ESA Hydroterra+ Earth Explorer 12 Phase 0 Study

## Contact

- Issues: [GitHub Issues](https://github.com/yannchemin/insar_soil_moisture/issues)
- Email: dr.yann.chemin@gmail.com
