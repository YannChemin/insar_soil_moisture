#!/usr/bin/env python3
"""
Basic example of InSAR Soil Moisture retrieval.

This example demonstrates how to use the package to process
a stack of Sentinel-1 SLC images and retrieve soil moisture.
"""

import os
import sys
import numpy as np
from datetime import datetime, timedelta

# Add parent directory to path for development
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from config import Config
from processor import InSARSoilMoistureProcessor
from postprocessing import PostProcessor
from io_utils import write_geotiff, save_metadata


def create_synthetic_data(n_images=20, rows=200, cols=300, seed=42):
    """
    Create synthetic SLC data for demonstration.

    In real usage, you would read actual Sentinel-1 SLC data.
    """
    np.random.seed(seed)

    print(f"Creating synthetic SLC stack: {n_images} images, {rows}x{cols} pixels")

    # Create coordinate grids
    x = np.linspace(0, 1, cols)
    y = np.linspace(0, 1, rows)
    X, Y = np.meshgrid(x, y)

    # Create base terrain phase (static)
    terrain_phase = 5 * np.sin(2 * np.pi * X) + 3 * np.cos(4 * np.pi * Y)

    # Create SLC stack with moisture-induced phase variations
    stack = np.zeros((n_images, rows, cols), dtype=np.complex64)

    for t in range(n_images):
        # Simulate seasonal soil moisture variation
        day_of_year = (t * 12) % 365
        seasonal = 0.3 + 0.2 * np.sin(2 * np.pi * day_of_year / 365)

        # Add spatial variation
        moisture = seasonal * (1 + 0.3 * np.sin(2 * np.pi * X) * np.cos(2 * np.pi * Y))

        # Moisture-induced phase (simplified model)
        moisture_phase = 2 * moisture

        # Total phase
        total_phase = terrain_phase + moisture_phase

        # Create complex SLC
        amplitude = 1 + 0.2 * np.random.randn(rows, cols)
        amplitude = np.clip(amplitude, 0.1, 2)

        stack[t] = amplitude * np.exp(1j * total_phase)

        # Add noise
        noise = 0.1 * (np.random.randn(rows, cols) + 1j * np.random.randn(rows, cols))
        stack[t] += noise

    # Create dates
    base_date = datetime(2020, 1, 1)
    dates = [base_date + timedelta(days=12 * i) for i in range(n_images)]

    # Create metadata
    metadata = {
        'geotransform': (500000, 10, 0, 4500000, 0, -10),  # UTM-like
        'projection': '+proj=utm +zone=32 +datum=WGS84',
        'rows': rows,
        'cols': cols,
        'dates': dates,
    }

    return stack, metadata


def main():
    """Run example processing."""
    print("=" * 60)
    print("InSAR Soil Moisture Retrieval - Basic Example")
    print("=" * 60)

    # Output directory
    output_dir = './example_output'
    os.makedirs(output_dir, exist_ok=True)

    # Create synthetic data (in real usage, read actual SLC data)
    stack, metadata = create_synthetic_data(
        n_images=20,
        rows=200,
        cols=300
    )

    print(f"\nStack shape: {stack.shape}")
    print(f"Date range: {metadata['dates'][0]} to {metadata['dates'][-1]}")

    # Configure processing
    config = Config(
        WINDOW_RANGE=30,          # Smaller window for demo
        WINDOW_AZIMUTH=20,
        TRIPLET_COHERENCE_THRESHOLD=0.01,  # Lower for synthetic data
        AGGREGATION_COHERENCE_THRESHOLD=0.05,
        OUTPUT_SPACING=500,       # 500m output
        TEMPORAL_FILTER_T=6,      # 6-day characteristic time
        SPATIAL_FILTER_SIGMA=1,   # Light spatial smoothing
    )

    print(f"\nConfiguration:")
    print(f"  Window: {config.WINDOW_RANGE} x {config.WINDOW_AZIMUTH} pixels")
    print(f"  Output spacing: {config.OUTPUT_SPACING} m")

    # Step 1: Core InSAR processing
    print("\n--- Core Processing ---")
    processor = InSARSoilMoistureProcessor(config)
    result = processor.process_stack(stack, metadata)

    print(f"Processed windows: {result.metadata['valid_windows']}")
    print(f"Mean coherence: {result.mean_coherence:.3f}")

    # Step 2: Post-processing
    print("\n--- Post-Processing ---")
    post = PostProcessor(config)

    moisture_cube, out_metadata = post.process(
        result.moisture_cube,
        result.coherence_map,
        metadata['dates'],
        metadata
    )

    print(f"Output shape: {moisture_cube.shape}")
    print(f"Valid pixels: {100 * (1 - np.mean(np.isnan(moisture_cube))):.1f}%")

    # Step 3: Save outputs
    print("\n--- Saving Outputs ---")

    # Save soil moisture time series
    sm_path = os.path.join(output_dir, 'soil_moisture.tif')
    write_geotiff(
        moisture_cube,
        sm_path,
        out_metadata.get('geotransform', metadata['geotransform']),
        metadata['projection'],
        band_names=[d.strftime('%Y-%m-%d') for d in metadata['dates']]
    )
    print(f"  Soil moisture: {sm_path}")

    # Save coherence map
    coh_path = os.path.join(output_dir, 'coherence.tif')
    write_geotiff(
        result.coherence_map,
        coh_path,
        out_metadata.get('geotransform', metadata['geotransform']),
        metadata['projection']
    )
    print(f"  Coherence: {coh_path}")

    # Save metadata
    meta_path = os.path.join(output_dir, 'metadata.json')
    save_metadata(
        meta_path,
        config,
        metadata['dates'],
        stats={
            'mean_coherence': float(result.mean_coherence),
            'valid_fraction': float(result.valid_fraction),
            'n_images': int(stack.shape[0]),
        }
    )
    print(f"  Metadata: {meta_path}")

    # Print summary statistics
    print("\n--- Summary Statistics ---")
    valid_data = moisture_cube[~np.isnan(moisture_cube)]
    print(f"  SM Index range: [{np.min(valid_data):.3f}, {np.max(valid_data):.3f}]")
    print(f"  SM Index mean: {np.mean(valid_data):.3f}")
    print(f"  SM Index std: {np.std(valid_data):.3f}")

    print("\n" + "=" * 60)
    print("Processing complete!")
    print(f"Outputs saved to: {os.path.abspath(output_dir)}")
    print("=" * 60)

    return moisture_cube, result


if __name__ == '__main__':
    main()
