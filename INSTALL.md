# Installation Guide

This guide provides detailed instructions for installing the InSAR Soil Moisture Retrieval package and its dependencies.

## System Requirements

### Minimum Requirements

- **OS**: Linux (Ubuntu 20.04+), macOS (10.15+), or Windows 10/11
- **Python**: 3.8 or higher
- **RAM**: 8 GB minimum, 16 GB recommended
- **Storage**: 10 GB for software, additional space for data
- **CPU**: Multi-core processor recommended

### Recommended Setup

- **OS**: Ubuntu 22.04 LTS
- **Python**: 3.10+
- **RAM**: 32 GB for large datasets
- **Storage**: SSD with 100+ GB free space
- **CPU**: 8+ cores for parallel processing

## Installation Methods

### Method 1: pip (Recommended)

```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # Linux/macOS
# venv\Scripts\activate   # Windows

# Install from PyPI (when available)
pip install insar-soil-moisture

# Or install from source
pip install git+https://github.com/yourusername/insar_soil_moisture.git
```

### Method 2: From Source

```bash
# Clone the repository
git clone https://github.com/yourusername/insar_soil_moisture.git
cd insar_soil_moisture

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .
```

### Method 3: Conda Environment

```bash
# Create conda environment
conda create -n insar_sm python=3.10
conda activate insar_sm

# Install GDAL via conda (recommended for GDAL)
conda install -c conda-forge gdal

# Install remaining dependencies
pip install -r requirements.txt

# Install package
pip install -e .
```

## Dependencies

### Core Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| numpy | >=1.21.0 | Array operations |
| scipy | >=1.7.0 | Scientific computing |
| gdal | >=3.0.0 | Geospatial I/O |

### Optional Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| matplotlib | >=3.5.0 | Visualization |
| rasterio | >=1.2.0 | Alternative raster I/O |
| dask | >=2022.1.0 | Parallel processing |
| xarray | >=0.20.0 | NetCDF support |
| pytest | >=7.0.0 | Testing |

## GDAL Installation

GDAL can be tricky to install. Here are platform-specific instructions:

### Ubuntu/Debian

```bash
# Install system GDAL
sudo apt-get update
sudo apt-get install -y \
    gdal-bin \
    libgdal-dev \
    python3-gdal

# Get GDAL version
gdal-config --version

# Install Python bindings matching system version
pip install GDAL==$(gdal-config --version)
```

### macOS

```bash
# Using Homebrew
brew install gdal

# Install Python bindings
pip install GDAL==$(gdal-config --version)
```

### Windows

```bash
# Option 1: Use conda (recommended)
conda install -c conda-forge gdal

# Option 2: Use pre-built wheels from Christoph Gohlke
# Download from: https://www.lfd.uci.edu/~gohlke/pythonlibs/#gdal
pip install GDAL-3.4.3-cp310-cp310-win_amd64.whl
```

### Using Conda (All Platforms)

```bash
# This is often the easiest method
conda install -c conda-forge gdal python-gdal
```

## Pre-processing Tools

The algorithm requires co-registered SLC data. Install one of these tools:

### ISCE2 (Recommended)

```bash
# Using conda
conda install -c conda-forge isce2

# Or from source
git clone https://github.com/isce-framework/isce2.git
cd isce2
python setup.py install
```

### ESA SNAP

1. Download from: https://step.esa.int/main/download/snap-download/
2. Install following the GUI installer
3. Configure Python integration:

```bash
# Add SNAP to PATH
export PATH=$PATH:/path/to/snap/bin

# Install snappy (SNAP Python interface)
cd /path/to/snap/.snap/snap-python/snappy
python setup.py install
```

### ASF Tools

```bash
# Install ASF tools for data download
pip install asf_search

# Configure credentials
asf_search --save-credentials
```

## Verification

After installation, verify everything works:

```bash
# Check Python version
python --version

# Check GDAL
python -c "from osgeo import gdal; print(f'GDAL version: {gdal.__version__}')"

# Check package installation
python -c "import insar_soil_moisture; print('Package installed successfully')"

# Run tests
pytest tests/
```

### Quick Test

```bash
# Run example with test data
python -c "
from insar_soil_moisture import Config, InSARSoilMoistureProcessor
config = Config()
print(f'Window size: {config.WINDOW_RANGE} x {config.WINDOW_AZIMUTH}')
print('Installation successful!')
"
```

## Troubleshooting

### GDAL Import Errors

**Error**: `ImportError: No module named 'osgeo'`

**Solution**:
```bash
# Check if GDAL is installed
pip show GDAL

# If not, install it
pip install GDAL==$(gdal-config --version)

# If gdal-config not found, install system GDAL first
sudo apt-get install libgdal-dev  # Ubuntu
brew install gdal                  # macOS
```

### Version Mismatch

**Error**: `GDAL version mismatch`

**Solution**:
```bash
# Uninstall current GDAL
pip uninstall GDAL

# Install matching version
pip install GDAL==$(gdal-config --version) --no-cache-dir
```

### Memory Errors

**Error**: `MemoryError during processing`

**Solution**:
```python
# Use tiled processing
processor.process(
    slc_paths=slc_list,
    tile_size=(500, 500),
    max_memory_gb=8
)
```

### Permission Errors

**Error**: `PermissionError: [Errno 13]`

**Solution**:
```bash
# Don't use sudo with pip
pip install --user insar-soil-moisture

# Or use virtual environment
python -m venv venv
source venv/bin/activate
pip install insar-soil-moisture
```

## Docker Installation

For reproducible environments, use Docker:

```dockerfile
# Dockerfile
FROM osgeo/gdal:ubuntu-small-3.6.0

RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

COPY . .
RUN pip3 install -e .

ENTRYPOINT ["insar-sm"]
```

Build and run:

```bash
# Build image
docker build -t insar-sm .

# Run processing
docker run -v /path/to/data:/data insar-sm \
    --slc-dir /data/slcs \
    --output-dir /data/output
```

## Cluster Installation

### SLURM Environment

```bash
# Module-based installation
module load python/3.10
module load gdal/3.4

# Create virtual environment in project space
python -m venv $PROJECT/venv/insar_sm
source $PROJECT/venv/insar_sm/bin/activate

# Install
pip install -r requirements.txt
pip install -e .
```

### PBS/Torque

```bash
#!/bin/bash
#PBS -N insar_install
#PBS -l nodes=1:ppn=1
#PBS -l walltime=00:30:00

module load anaconda3
conda create -n insar_sm python=3.10 gdal -c conda-forge -y
conda activate insar_sm
pip install -e /path/to/insar_soil_moisture
```

## Updating

```bash
# Update from PyPI
pip install --upgrade insar-soil-moisture

# Update from source
cd insar_soil_moisture
git pull
pip install -e . --upgrade
```

## Uninstalling

```bash
# Uninstall package
pip uninstall insar-soil-moisture

# Remove virtual environment
rm -rf venv/
```

## Getting Help

If you encounter issues:

1. Check the [FAQ](docs/FAQ.md)
2. Search [GitHub Issues](https://github.com/yourusername/insar_soil_moisture/issues)
3. Open a new issue with:
   - Operating system and version
   - Python version (`python --version`)
   - GDAL version (`gdalinfo --version`)
   - Full error traceback
   - Steps to reproduce
