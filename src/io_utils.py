"""
I/O utilities for InSAR Soil Moisture processing.

This module provides functions for reading and writing geospatial data,
including SLC stacks and output rasters.
"""

import os
import json
import logging
from datetime import datetime
from typing import List, Tuple, Optional, Dict, Any, Union
import re

import numpy as np

try:
    from osgeo import gdal, osr
    gdal.UseExceptions()
except ImportError:
    raise ImportError(
        "GDAL is required but not installed. "
        "Install with: pip install GDAL==$(gdal-config --version)"
    )

logger = logging.getLogger(__name__)


def read_slc_stack(
    slc_paths: List[str],
    band_real: int = 1,
    band_imag: int = 2,
    dtype: np.dtype = np.complex64
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Read a stack of co-registered SLC images.

    Supports both complex single-band and dual-band (I/Q) formats.

    Args:
        slc_paths: List of paths to co-registered SLC GeoTIFFs
        band_real: Band number for real component (default: 1)
        band_imag: Band number for imaginary component (default: 2)
        dtype: Output data type (default: complex64)

    Returns:
        stack: Complex array of shape (N, rows, cols)
        metadata: Dictionary with geotransform, projection, dates, etc.

    Raises:
        FileNotFoundError: If any SLC file is not found
        ValueError: If SLC dimensions don't match

    Example:
        >>> stack, meta = read_slc_stack(['slc_001.tif', 'slc_002.tif'])
        >>> print(stack.shape)
        (2, 1000, 2000)
    """
    if not slc_paths:
        raise ValueError("slc_paths cannot be empty")

    # Verify all files exist
    for path in slc_paths:
        if not os.path.exists(path):
            raise FileNotFoundError(f"SLC file not found: {path}")

    logger.info(f"Reading {len(slc_paths)} SLC images...")

    # Open first file to get dimensions and metadata
    ds0 = gdal.Open(slc_paths[0], gdal.GA_ReadOnly)
    if ds0 is None:
        raise IOError(f"Could not open: {slc_paths[0]}")

    rows = ds0.RasterYSize
    cols = ds0.RasterXSize
    n_bands = ds0.RasterCount
    n_images = len(slc_paths)

    # Determine if complex or I/Q format
    is_complex = n_bands == 1 and ds0.GetRasterBand(1).DataType in [
        gdal.GDT_CFloat32, gdal.GDT_CFloat64, gdal.GDT_CInt16, gdal.GDT_CInt32
    ]

    logger.debug(f"Image dimensions: {rows} x {cols}, bands: {n_bands}, complex: {is_complex}")

    # Allocate stack
    stack = np.zeros((n_images, rows, cols), dtype=dtype)

    # Read all images
    for i, path in enumerate(slc_paths):
        ds = gdal.Open(path, gdal.GA_ReadOnly)
        if ds is None:
            raise IOError(f"Could not open: {path}")

        # Check dimensions match
        if ds.RasterYSize != rows or ds.RasterXSize != cols:
            raise ValueError(
                f"Dimension mismatch for {path}: "
                f"expected ({rows}, {cols}), got ({ds.RasterYSize}, {ds.RasterXSize})"
            )

        if is_complex or ds.RasterCount == 1:
            # Single complex band
            data = ds.GetRasterBand(1).ReadAsArray()
            if np.isrealobj(data):
                # Real data, convert to complex
                stack[i] = data.astype(dtype)
            else:
                stack[i] = data
        else:
            # I/Q format (separate real and imaginary bands)
            real = ds.GetRasterBand(band_real).ReadAsArray().astype(np.float32)
            imag = ds.GetRasterBand(band_imag).ReadAsArray().astype(np.float32)
            stack[i] = real + 1j * imag

        ds = None

        if (i + 1) % 10 == 0:
            logger.debug(f"Read {i + 1}/{n_images} images")

    # Collect metadata
    metadata = {
        'geotransform': ds0.GetGeoTransform(),
        'projection': ds0.GetProjection(),
        'rows': rows,
        'cols': cols,
        'n_images': n_images,
        'dates': extract_dates_from_paths(slc_paths),
        'paths': slc_paths,
    }

    ds0 = None

    logger.info(f"Loaded stack with shape: {stack.shape}")
    return stack, metadata


def extract_dates_from_paths(paths: List[str]) -> List[Optional[datetime]]:
    """
    Extract acquisition dates from Sentinel-1 filenames.

    Supports multiple naming conventions:
    - Standard: S1A_IW_SLC__1SDV_YYYYMMDDTHHMMSS_...
    - Simplified: YYYYMMDD_slc.tif
    - Custom: *_YYYYMMDD_*.tif

    Args:
        paths: List of file paths

    Returns:
        List of datetime objects (None for unrecognized formats)
    """
    dates = []

    # Patterns to try
    patterns = [
        r'S1[AB]_\w+_(\d{8})T\d{6}',  # Standard Sentinel-1
        r'(\d{8})_slc',                # Simplified
        r'_(\d{8})_',                  # Generic YYYYMMDD
        r'^(\d{8})',                   # Starts with date
    ]

    for path in paths:
        basename = os.path.basename(path)
        date_found = None

        for pattern in patterns:
            match = re.search(pattern, basename)
            if match:
                try:
                    date_str = match.group(1)
                    date_found = datetime.strptime(date_str, '%Y%m%d')
                    break
                except ValueError:
                    continue

        dates.append(date_found)

        if date_found is None:
            logger.warning(f"Could not extract date from: {basename}")

    return dates


def write_geotiff(
    data: np.ndarray,
    output_path: str,
    geotransform: Tuple[float, ...],
    projection: str,
    nodata: float = np.nan,
    compress: str = "LZW",
    band_names: Optional[List[str]] = None,
    metadata: Optional[Dict[str, str]] = None
) -> None:
    """
    Write array to GeoTIFF file.

    Args:
        data: 2D or 3D array (bands, rows, cols)
        output_path: Output file path
        geotransform: GDAL geotransform tuple
        projection: WKT projection string
        nodata: NoData value (default: NaN)
        compress: Compression algorithm (LZW, DEFLATE, ZSTD, NONE)
        band_names: Optional list of band names/descriptions
        metadata: Optional dictionary of metadata to attach

    Raises:
        ValueError: If data has invalid dimensions
    """
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)

    # Handle dimensions
    if data.ndim == 2:
        bands = 1
        rows, cols = data.shape
        data = data[np.newaxis, :, :]
    elif data.ndim == 3:
        bands, rows, cols = data.shape
    else:
        raise ValueError(f"Data must be 2D or 3D, got {data.ndim}D")

    # Map numpy dtype to GDAL dtype
    dtype_map = {
        np.float32: gdal.GDT_Float32,
        np.float64: gdal.GDT_Float64,
        np.int16: gdal.GDT_Int16,
        np.int32: gdal.GDT_Int32,
        np.uint8: gdal.GDT_Byte,
        np.uint16: gdal.GDT_UInt16,
        np.uint32: gdal.GDT_UInt32,
    }

    gdal_dtype = dtype_map.get(data.dtype.type, gdal.GDT_Float32)

    # Create output dataset
    driver = gdal.GetDriverByName('GTiff')
    options = ['TILED=YES', f'COMPRESS={compress}', 'BIGTIFF=IF_SAFER']

    ds = driver.Create(output_path, cols, rows, bands, gdal_dtype, options=options)

    if ds is None:
        raise IOError(f"Could not create output file: {output_path}")

    ds.SetGeoTransform(geotransform)
    ds.SetProjection(projection)

    # Set metadata
    if metadata:
        for key, value in metadata.items():
            ds.SetMetadataItem(key, str(value))

    # Write bands
    for b in range(bands):
        band = ds.GetRasterBand(b + 1)
        band.WriteArray(data[b].astype(np.float32))
        band.SetNoDataValue(float(nodata) if not np.isnan(nodata) else -9999.0)

        if band_names and b < len(band_names):
            band.SetDescription(band_names[b])

    ds.FlushCache()
    ds = None

    logger.info(f"Wrote {output_path} ({bands} bands, {rows}x{cols})")


def write_netcdf(
    data: np.ndarray,
    output_path: str,
    dates: List[datetime],
    geotransform: Tuple[float, ...],
    projection: str,
    variable_name: str = "soil_moisture",
    attributes: Optional[Dict[str, Any]] = None
) -> None:
    """
    Write data to NetCDF file with CF conventions.

    Args:
        data: 3D array (time, y, x)
        output_path: Output file path
        dates: List of dates for time dimension
        geotransform: GDAL geotransform tuple
        projection: WKT projection string
        variable_name: Name of the main variable
        attributes: Optional global attributes
    """
    try:
        import netCDF4 as nc
    except ImportError:
        raise ImportError("netCDF4 is required for NetCDF output. Install with: pip install netCDF4")

    n_times, n_rows, n_cols = data.shape

    # Calculate coordinates from geotransform
    x_origin, x_res, _, y_origin, _, y_res = geotransform
    x_coords = x_origin + (np.arange(n_cols) + 0.5) * x_res
    y_coords = y_origin + (np.arange(n_rows) + 0.5) * y_res

    # Create NetCDF file
    ds = nc.Dataset(output_path, 'w', format='NETCDF4')

    # Create dimensions
    ds.createDimension('time', n_times)
    ds.createDimension('y', n_rows)
    ds.createDimension('x', n_cols)

    # Create coordinate variables
    time_var = ds.createVariable('time', 'f8', ('time',))
    time_var.units = f'days since {dates[0].strftime("%Y-%m-%d")}'
    time_var.calendar = 'standard'
    time_var[:] = [(d - dates[0]).days for d in dates]

    x_var = ds.createVariable('x', 'f8', ('x',))
    x_var.units = 'meters'
    x_var.standard_name = 'projection_x_coordinate'
    x_var[:] = x_coords

    y_var = ds.createVariable('y', 'f8', ('y',))
    y_var.units = 'meters'
    y_var.standard_name = 'projection_y_coordinate'
    y_var[:] = y_coords

    # Create data variable
    sm_var = ds.createVariable(
        variable_name, 'f4', ('time', 'y', 'x'),
        fill_value=-9999.0, zlib=True, complevel=4
    )
    sm_var.long_name = 'Surface Soil Moisture Index'
    sm_var.units = '1'
    sm_var.valid_range = [0.0, 1.0]
    sm_var[:] = data

    # Add CRS
    crs_var = ds.createVariable('crs', 'i4')
    crs_var.grid_mapping_name = 'transverse_mercator'
    crs_var.crs_wkt = projection

    # Global attributes
    ds.Conventions = 'CF-1.8'
    ds.title = 'InSAR-derived Surface Soil Moisture'
    ds.source = 'Sentinel-1 SAR Interferometry'
    ds.history = f'Created {datetime.now().isoformat()}'

    if attributes:
        for key, value in attributes.items():
            setattr(ds, key, value)

    ds.close()
    logger.info(f"Wrote {output_path}")


def save_metadata(
    output_path: str,
    config: Any,
    dates: List[datetime],
    stats: Optional[Dict[str, Any]] = None
) -> None:
    """
    Save processing metadata to JSON file.

    Args:
        output_path: Output file path
        config: Configuration object
        dates: List of acquisition dates
        stats: Optional processing statistics
    """
    metadata = {
        'processing_date': datetime.now().isoformat(),
        'software_version': '1.0.0',
        'algorithm': 'InSAR Closure Phase Soil Moisture (De Zan et al. 2026)',
        'configuration': config.to_dict() if hasattr(config, 'to_dict') else str(config),
        'acquisition_dates': [d.isoformat() if d else None for d in dates],
        'n_acquisitions': len(dates),
    }

    if stats:
        metadata['statistics'] = stats

    with open(output_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Saved metadata to {output_path}")


def get_raster_info(raster_path: str) -> Dict[str, Any]:
    """
    Get information about a raster file.

    Args:
        raster_path: Path to raster file

    Returns:
        Dictionary with raster metadata
    """
    ds = gdal.Open(raster_path, gdal.GA_ReadOnly)
    if ds is None:
        raise IOError(f"Could not open: {raster_path}")

    gt = ds.GetGeoTransform()

    info = {
        'path': raster_path,
        'driver': ds.GetDriver().ShortName,
        'width': ds.RasterXSize,
        'height': ds.RasterYSize,
        'bands': ds.RasterCount,
        'dtype': gdal.GetDataTypeName(ds.GetRasterBand(1).DataType),
        'projection': ds.GetProjection(),
        'geotransform': gt,
        'pixel_size': (abs(gt[1]), abs(gt[5])),
        'extent': {
            'xmin': gt[0],
            'xmax': gt[0] + gt[1] * ds.RasterXSize,
            'ymin': gt[3] + gt[5] * ds.RasterYSize,
            'ymax': gt[3],
        },
    }

    ds = None
    return info


def create_vrt_stack(
    slc_paths: List[str],
    output_vrt: str
) -> str:
    """
    Create a VRT (Virtual Raster) stack from multiple SLC files.

    This is memory-efficient for large datasets.

    Args:
        slc_paths: List of input SLC paths
        output_vrt: Output VRT file path

    Returns:
        Path to created VRT file
    """
    vrt_options = gdal.BuildVRTOptions(
        separate=True,  # Each input as separate band
        resolution='highest'
    )

    vrt = gdal.BuildVRT(output_vrt, slc_paths, options=vrt_options)

    if vrt is None:
        raise IOError(f"Could not create VRT: {output_vrt}")

    vrt = None
    logger.info(f"Created VRT stack: {output_vrt}")

    return output_vrt


def reproject_raster(
    input_path: str,
    output_path: str,
    target_crs: str,
    resolution: Optional[Tuple[float, float]] = None,
    resampling: str = 'bilinear'
) -> None:
    """
    Reproject raster to new CRS.

    Args:
        input_path: Input raster path
        output_path: Output raster path
        target_crs: Target CRS (EPSG code or WKT)
        resolution: Target resolution (x, y) in target CRS units
        resampling: Resampling method (nearest, bilinear, cubic, etc.)
    """
    resampling_map = {
        'nearest': gdal.GRA_NearestNeighbour,
        'bilinear': gdal.GRA_Bilinear,
        'cubic': gdal.GRA_Cubic,
        'cubicspline': gdal.GRA_CubicSpline,
        'lanczos': gdal.GRA_Lanczos,
        'average': gdal.GRA_Average,
    }

    warp_options = gdal.WarpOptions(
        dstSRS=target_crs,
        xRes=resolution[0] if resolution else None,
        yRes=resolution[1] if resolution else None,
        resampleAlg=resampling_map.get(resampling, gdal.GRA_Bilinear),
        format='GTiff',
        creationOptions=['COMPRESS=LZW', 'TILED=YES']
    )

    gdal.Warp(output_path, input_path, options=warp_options)
    logger.info(f"Reprojected {input_path} to {output_path}")
