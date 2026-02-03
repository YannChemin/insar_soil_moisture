"""
Command-line interface for InSAR Soil Moisture retrieval.

This module provides the main CLI entry point for processing
Sentinel-1 data to retrieve soil moisture.
"""

import os
import sys
import logging
from datetime import datetime
from typing import Optional, List
import json

import click
import numpy as np

from .config import Config, create_example_config
from .io_utils import read_slc_stack, write_geotiff, save_metadata
from .processor import InSARSoilMoistureProcessor
from .postprocessing import PostProcessor


def setup_logging(level: str, log_file: Optional[str] = None):
    """Configure logging."""
    numeric_level = getattr(logging, level.upper(), logging.INFO)

    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=numeric_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )


@click.group()
@click.version_option(version='1.0.0')
def cli():
    """
    InSAR Soil Moisture Retrieval Tool

    Process Sentinel-1 SAR data to retrieve high-resolution soil moisture
    using interferometric closure phases.

    Reference:
        De Zan et al. (2026). Validation of high-resolution surface soil
        moisture time series retrieved by means of SAR interferometry.
        Remote Sensing of Environment.
    """
    pass


@cli.command()
@click.option('--slc-dir', '-s', required=True, type=click.Path(exists=True),
              help='Directory containing co-registered SLC GeoTIFFs')
@click.option('--output-dir', '-o', required=True, type=click.Path(),
              help='Output directory')
@click.option('--config', '-c', type=click.Path(exists=True),
              help='Configuration YAML file')
@click.option('--start-date', type=click.DateTime(formats=['%Y-%m-%d']),
              help='Start date for processing (YYYY-MM-DD)')
@click.option('--end-date', type=click.DateTime(formats=['%Y-%m-%d']),
              help='End date for processing (YYYY-MM-DD)')
@click.option('--no-temporal-filter', is_flag=True,
              help='Disable temporal filtering')
@click.option('--no-spatial-filter', is_flag=True,
              help='Disable spatial filtering')
@click.option('--output-format', type=click.Choice(['GTiff', 'NetCDF']),
              default='GTiff', help='Output format')
@click.option('--parallel', '-p', type=int, default=1,
              help='Number of parallel workers')
@click.option('--log-level', type=click.Choice(['DEBUG', 'INFO', 'WARNING', 'ERROR']),
              default='INFO', help='Logging level')
@click.option('--log-file', type=click.Path(),
              help='Log file path')
def process(
    slc_dir: str,
    output_dir: str,
    config: Optional[str],
    start_date: Optional[datetime],
    end_date: Optional[datetime],
    no_temporal_filter: bool,
    no_spatial_filter: bool,
    output_format: str,
    parallel: int,
    log_level: str,
    log_file: Optional[str]
):
    """
    Process SLC stack to retrieve soil moisture.

    Example:
        insar-sm process --slc-dir ./slcs --output-dir ./output
    """
    setup_logging(log_level, log_file)
    logger = logging.getLogger(__name__)

    logger.info("=" * 60)
    logger.info("InSAR Soil Moisture Retrieval")
    logger.info("=" * 60)

    # Load configuration
    if config:
        cfg = Config.from_yaml(config)
    else:
        cfg = Config()

    # Override config with CLI options
    if no_temporal_filter:
        cfg.TEMPORAL_FILTER_ENABLED = False
    if no_spatial_filter:
        cfg.SPATIAL_FILTER_ENABLED = False
    if parallel > 0:
        cfg.N_PARALLEL = parallel

    logger.info(f"\n{cfg}")

    # Find SLC files
    slc_files = sorted([
        os.path.join(slc_dir, f)
        for f in os.listdir(slc_dir)
        if f.lower().endswith(('.tif', '.tiff'))
    ])

    if len(slc_files) < 3:
        logger.error(f"Need at least 3 SLC images, found {len(slc_files)}")
        sys.exit(1)

    logger.info(f"Found {len(slc_files)} SLC files")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    try:
        # Read SLC stack
        logger.info("Reading SLC stack...")
        stack, metadata = read_slc_stack(slc_files)

        # Filter by date if specified
        if start_date or end_date:
            stack, metadata = filter_by_date(
                stack, metadata, start_date, end_date
            )

        # Core processing
        logger.info("Starting core InSAR processing...")
        processor = InSARSoilMoistureProcessor(cfg)
        result = processor.process_stack(stack, metadata)

        # Post-processing
        logger.info("Applying post-processing...")
        post = PostProcessor(cfg)
        moisture_cube, out_metadata = post.process(
            result.moisture_cube,
            result.coherence_map,
            metadata.get('dates', []),
            metadata
        )

        # Save outputs
        logger.info("Saving outputs...")

        output_path = os.path.join(output_dir, 'insar_soil_moisture.tif')
        write_geotiff(
            moisture_cube,
            output_path,
            out_metadata.get('geotransform', metadata['geotransform']),
            metadata['projection'],
            nodata=cfg.NODATA_VALUE,
            compress=cfg.COMPRESS
        )

        # Save coherence map
        coh_path = os.path.join(output_dir, 'coherence_map.tif')
        write_geotiff(
            result.coherence_map,
            coh_path,
            out_metadata.get('geotransform', metadata['geotransform']),
            metadata['projection']
        )

        # Save metadata
        meta_path = os.path.join(output_dir, 'metadata.json')
        save_metadata(
            meta_path,
            cfg,
            metadata.get('dates', []),
            stats={
                'mean_coherence': result.mean_coherence,
                'valid_fraction': result.valid_fraction,
            }
        )

        logger.info("=" * 60)
        logger.info("Processing complete!")
        logger.info(f"Output: {output_path}")
        logger.info(f"Mean coherence: {result.mean_coherence:.3f}")
        logger.info(f"Valid pixels: {result.valid_fraction*100:.1f}%")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"Processing failed: {e}", exc_info=True)
        sys.exit(1)


@cli.command()
@click.option('--output', '-o', required=True, type=click.Path(),
              help='Output configuration file path')
def init_config(output: str):
    """
    Create example configuration file.

    Example:
        insar-sm init-config --output config.yaml
    """
    create_example_config(output)
    click.echo(f"Created example configuration: {output}")


@cli.command()
@click.argument('raster_path', type=click.Path(exists=True))
def info(raster_path: str):
    """
    Display information about a raster file.

    Example:
        insar-sm info soil_moisture.tif
    """
    from .io_utils import get_raster_info

    info = get_raster_info(raster_path)

    click.echo(f"\nRaster Information: {info['path']}")
    click.echo("-" * 50)
    click.echo(f"Driver: {info['driver']}")
    click.echo(f"Size: {info['width']} x {info['height']} pixels")
    click.echo(f"Bands: {info['bands']}")
    click.echo(f"Data type: {info['dtype']}")
    click.echo(f"Pixel size: {info['pixel_size'][0]:.2f} x {info['pixel_size'][1]:.2f}")
    click.echo(f"Extent:")
    click.echo(f"  X: {info['extent']['xmin']:.2f} to {info['extent']['xmax']:.2f}")
    click.echo(f"  Y: {info['extent']['ymin']:.2f} to {info['extent']['ymax']:.2f}")


@cli.command()
@click.option('--slc-dir', '-s', required=True, type=click.Path(exists=True),
              help='Directory containing SLC files')
@click.option('--output', '-o', required=True, type=click.Path(),
              help='Output VRT file path')
def create_stack(slc_dir: str, output: str):
    """
    Create VRT stack from SLC files.

    Example:
        insar-sm create-stack --slc-dir ./slcs --output stack.vrt
    """
    from .io_utils import create_vrt_stack

    slc_files = sorted([
        os.path.join(slc_dir, f)
        for f in os.listdir(slc_dir)
        if f.lower().endswith(('.tif', '.tiff'))
    ])

    if not slc_files:
        click.echo("No SLC files found", err=True)
        sys.exit(1)

    create_vrt_stack(slc_files, output)
    click.echo(f"Created VRT stack: {output}")
    click.echo(f"Contains {len(slc_files)} bands")


@cli.command()
@click.argument('moisture_file', type=click.Path(exists=True))
@click.option('--output', '-o', type=click.Path(), help='Output plot file')
@click.option('--band', '-b', type=int, default=1, help='Band to plot')
def plot(moisture_file: str, output: Optional[str], band: int):
    """
    Quick visualization of soil moisture data.

    Example:
        insar-sm plot soil_moisture.tif --band 10
    """
    try:
        import matplotlib.pyplot as plt
        from osgeo import gdal
    except ImportError:
        click.echo("matplotlib is required for plotting", err=True)
        sys.exit(1)

    ds = gdal.Open(moisture_file)
    if ds is None:
        click.echo(f"Could not open {moisture_file}", err=True)
        sys.exit(1)

    if band > ds.RasterCount:
        click.echo(f"Band {band} not found (file has {ds.RasterCount} bands)", err=True)
        sys.exit(1)

    data = ds.GetRasterBand(band).ReadAsArray()
    ds = None

    # Replace nodata with NaN
    data = data.astype(float)
    data[data < -9000] = np.nan

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(data, cmap='YlGnBu', vmin=0, vmax=1)
    ax.set_title(f'Soil Moisture Index - Band {band}')
    plt.colorbar(im, ax=ax, label='SM Index')

    if output:
        plt.savefig(output, dpi=150, bbox_inches='tight')
        click.echo(f"Saved plot to {output}")
    else:
        plt.show()


def filter_by_date(
    stack: np.ndarray,
    metadata: dict,
    start_date: Optional[datetime],
    end_date: Optional[datetime]
) -> tuple:
    """Filter stack by date range."""
    dates = metadata.get('dates', [])

    if not dates:
        return stack, metadata

    mask = np.ones(len(dates), dtype=bool)

    for i, d in enumerate(dates):
        if d is None:
            continue
        if start_date and d < start_date:
            mask[i] = False
        if end_date and d > end_date:
            mask[i] = False

    filtered_stack = stack[mask]
    filtered_dates = [d for d, m in zip(dates, mask) if m]

    new_metadata = metadata.copy()
    new_metadata['dates'] = filtered_dates
    new_metadata['n_images'] = len(filtered_dates)

    return filtered_stack, new_metadata


def main():
    """Main entry point."""
    cli()


if __name__ == '__main__':
    main()
