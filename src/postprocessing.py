"""
Post-processing module for InSAR Soil Moisture data.

This module provides functions for:
- Sign correction
- Temporal filtering (exponential filter)
- Spatial filtering (Gaussian smoothing)
- Grid aggregation
- Quality masking
"""

import numpy as np
from scipy import ndimage
from scipy.stats import spearmanr
from typing import List, Tuple, Optional, Dict, Any
from datetime import datetime
import logging

from .config import Config

logger = logging.getLogger(__name__)


class PostProcessor:
    """
    Post-processing for InSAR soil moisture products.

    Implements the post-processing steps from De Zan et al. (2026):
    1. Sign correction based on correlation with scene average
    2. Coherence-based quality masking
    3. Temporal filtering (exponential filter)
    4. Spatial filtering (Gaussian smoothing)
    5. Aggregation to output grid
    """

    def __init__(self, config: Optional[Config] = None):
        """
        Initialize post-processor.

        Args:
            config: Processing configuration
        """
        self.config = config or Config()

    def process(
        self,
        moisture_cube: np.ndarray,
        coherence_map: np.ndarray,
        dates: List[datetime],
        metadata: Dict[str, Any]
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Apply full post-processing pipeline.

        Args:
            moisture_cube: Raw moisture time series (n_dates, rows, cols)
            coherence_map: Mean coherence map
            dates: Acquisition dates
            metadata: Processing metadata

        Returns:
            Processed moisture cube and updated metadata
        """
        # Step 1: Sign correction
        moisture_cube = self.correct_sign(moisture_cube)

        # Step 2: Coherence masking
        moisture_cube = self.apply_coherence_mask(moisture_cube, coherence_map)

        # Step 3: Temporal filtering
        if self.config.TEMPORAL_FILTER_ENABLED:
            moisture_cube = self.temporal_filter(moisture_cube, dates)

        # Step 4: Spatial filtering
        if self.config.SPATIAL_FILTER_ENABLED:
            moisture_cube = self.spatial_filter(moisture_cube)

        # Step 5: Grid aggregation
        moisture_cube, metadata = self.aggregate_to_grid(moisture_cube, metadata)

        # Normalize to 0-1 range
        moisture_cube = self.normalize(moisture_cube)

        return moisture_cube, metadata

    def correct_sign(
        self,
        moisture_cube: np.ndarray,
        reference: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Correct sign ambiguity in moisture time series.

        The InSAR algorithm may produce inverted moisture signals
        due to the model assumption about scattering profiles.
        This corrects by correlating with the scene average.

        Args:
            moisture_cube: Moisture time series (n_dates, rows, cols)
            reference: Optional reference time series for correlation

        Returns:
            Sign-corrected moisture cube
        """
        n_dates, n_row, n_col = moisture_cube.shape

        # Compute scene average (excluding NaN)
        if reference is None:
            reference = np.nanmean(moisture_cube, axis=(1, 2))

        corrected = moisture_cube.copy()
        n_flipped = 0

        for r in range(n_row):
            for c in range(n_col):
                ts = moisture_cube[:, r, c]

                if np.all(np.isnan(ts)):
                    continue

                # Compute Spearman correlation with reference
                valid = ~np.isnan(ts) & ~np.isnan(reference)
                if np.sum(valid) < 5:
                    continue

                corr, _ = spearmanr(ts[valid], reference[valid])

                if corr < 0:
                    corrected[:, r, c] = -ts
                    n_flipped += 1

        valid_pixels = np.sum(~np.all(np.isnan(moisture_cube), axis=0))
        logger.info(f"Sign correction: flipped {n_flipped}/{valid_pixels} pixels "
                   f"({100*n_flipped/max(1,valid_pixels):.1f}%)")

        return corrected

    def apply_coherence_mask(
        self,
        moisture_cube: np.ndarray,
        coherence_map: np.ndarray
    ) -> np.ndarray:
        """
        Mask low-coherence pixels.

        Args:
            moisture_cube: Moisture time series
            coherence_map: Mean coherence map

        Returns:
            Masked moisture cube (low coherence pixels set to NaN)
        """
        mask = coherence_map < self.config.AGGREGATION_COHERENCE_THRESHOLD

        masked = moisture_cube.copy()
        masked[:, mask] = np.nan

        n_masked = np.sum(mask)
        total = mask.size
        logger.info(f"Coherence masking: removed {n_masked}/{total} pixels "
                   f"({100*n_masked/total:.1f}%)")

        return masked

    def temporal_filter(
        self,
        moisture_cube: np.ndarray,
        dates: List[datetime],
        method: str = 'exponential'
    ) -> np.ndarray:
        """
        Apply temporal filtering to smooth time series.

        Implements the exponential filter (SWI-like) from Brocca et al. (2010):
        SM_filtered(t) = Σ SM(ti) * exp(-(t-ti)/T) / Σ exp(-(t-ti)/T)

        Args:
            moisture_cube: Moisture time series (n_dates, rows, cols)
            dates: Acquisition dates
            method: Filter method ('exponential', 'moving_average')

        Returns:
            Temporally filtered moisture cube
        """
        if method == 'moving_average':
            return self._moving_average_filter(moisture_cube, dates)

        # Exponential filter
        T = self.config.TEMPORAL_FILTER_T
        n_dates, n_row, n_col = moisture_cube.shape

        filtered = np.full_like(moisture_cube, np.nan)

        # Convert dates to days from start
        valid_dates = [d for d in dates if d is not None]
        if not valid_dates:
            logger.warning("No valid dates for temporal filtering")
            return moisture_cube

        ref_date = min(valid_dates)
        days = np.array([(d - ref_date).days if d else np.nan for d in dates])

        for t in range(n_dates):
            if np.isnan(days[t]):
                continue

            # Compute weights for all previous acquisitions
            weights = np.zeros(n_dates)
            for ti in range(t + 1):
                if np.isnan(days[ti]):
                    continue
                dt = days[t] - days[ti]
                weights[ti] = np.exp(-dt / T)

            # Apply weighted average pixel by pixel
            for r in range(n_row):
                for c in range(n_col):
                    ts = moisture_cube[:t + 1, r, c]
                    valid = ~np.isnan(ts)

                    if np.sum(valid) == 0:
                        continue

                    w = weights[:t + 1][valid]
                    w_sum = np.sum(w)

                    if w_sum > 0:
                        filtered[t, r, c] = np.sum(ts[valid] * w) / w_sum

        logger.info(f"Applied exponential temporal filter (T={T} days)")
        return filtered

    def _moving_average_filter(
        self,
        moisture_cube: np.ndarray,
        dates: List[datetime],
        window_days: int = 10
    ) -> np.ndarray:
        """Simple moving average filter."""
        n_dates, n_row, n_col = moisture_cube.shape
        filtered = np.full_like(moisture_cube, np.nan)

        ref_date = min(d for d in dates if d)
        days = np.array([(d - ref_date).days if d else np.nan for d in dates])

        for t in range(n_dates):
            if np.isnan(days[t]):
                continue

            # Find acquisitions within window
            mask = (days >= days[t] - window_days) & (days <= days[t])

            if np.sum(mask) > 0:
                filtered[t] = np.nanmean(moisture_cube[mask], axis=0)

        return filtered

    def spatial_filter(
        self,
        moisture_cube: np.ndarray
    ) -> np.ndarray:
        """
        Apply 2D Gaussian spatial smoothing.

        Args:
            moisture_cube: Moisture time series

        Returns:
            Spatially filtered moisture cube
        """
        sigma = self.config.SPATIAL_FILTER_SIGMA
        n_dates = moisture_cube.shape[0]

        filtered = np.zeros_like(moisture_cube)

        for t in range(n_dates):
            img = moisture_cube[t]

            # Handle NaN values
            mask = np.isnan(img)
            img_filled = np.where(mask, 0, img)

            # Apply Gaussian filter
            smoothed = ndimage.gaussian_filter(img_filled, sigma)

            # Normalize by valid pixel contribution
            weight = ndimage.gaussian_filter((~mask).astype(float), sigma)
            weight[weight < 0.1] = np.nan

            filtered[t] = smoothed / weight
            filtered[t, mask & np.isnan(weight)] = np.nan

        logger.info(f"Applied Gaussian spatial filter (sigma={sigma} pixels)")
        return filtered

    def aggregate_to_grid(
        self,
        moisture_cube: np.ndarray,
        metadata: Dict[str, Any]
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Aggregate to regular output grid.

        Args:
            moisture_cube: Moisture time series
            metadata: Processing metadata

        Returns:
            Aggregated moisture cube and updated metadata
        """
        gt = metadata.get('geotransform', (0, 1, 0, 0, 0, -1))
        pixel_size = abs(gt[1])

        # Calculate aggregation factor
        current_spacing = pixel_size * self.config.WINDOW_RANGE
        target_spacing = self.config.OUTPUT_SPACING
        agg_factor = int(np.ceil(target_spacing / current_spacing))

        if agg_factor <= 1:
            logger.info("No aggregation needed")
            return moisture_cube, metadata

        n_dates, n_row, n_col = moisture_cube.shape
        new_rows = n_row // agg_factor
        new_cols = n_col // agg_factor

        if new_rows == 0 or new_cols == 0:
            logger.warning("Aggregation would result in empty grid")
            return moisture_cube, metadata

        aggregated = np.full((n_dates, new_rows, new_cols), np.nan, dtype=np.float32)

        for t in range(n_dates):
            for r in range(new_rows):
                for c in range(new_cols):
                    block = moisture_cube[
                        t,
                        r * agg_factor:(r + 1) * agg_factor,
                        c * agg_factor:(c + 1) * agg_factor
                    ]
                    if not np.all(np.isnan(block)):
                        aggregated[t, r, c] = np.nanmean(block)

        # Update geotransform
        new_gt = list(gt)
        new_gt[1] *= agg_factor * self.config.WINDOW_RANGE
        new_gt[5] *= agg_factor * self.config.WINDOW_AZIMUTH

        new_metadata = metadata.copy()
        new_metadata['geotransform'] = tuple(new_gt)
        new_metadata['rows'] = new_rows
        new_metadata['cols'] = new_cols
        new_metadata['aggregation_factor'] = agg_factor

        logger.info(f"Aggregated from {n_row}x{n_col} to {new_rows}x{new_cols} "
                   f"(factor {agg_factor})")

        return aggregated, new_metadata

    def normalize(
        self,
        moisture_cube: np.ndarray,
        method: str = 'percentile',
        percentiles: Tuple[float, float] = (2, 98)
    ) -> np.ndarray:
        """
        Normalize moisture values to 0-1 range.

        Args:
            moisture_cube: Moisture time series
            method: Normalization method ('percentile', 'minmax', 'zscore')
            percentiles: Percentiles for clipping (if method='percentile')

        Returns:
            Normalized moisture cube
        """
        valid = ~np.isnan(moisture_cube)

        if not np.any(valid):
            return moisture_cube

        if method == 'percentile':
            vmin, vmax = np.nanpercentile(moisture_cube, percentiles)
        elif method == 'minmax':
            vmin, vmax = np.nanmin(moisture_cube), np.nanmax(moisture_cube)
        else:  # zscore
            mean = np.nanmean(moisture_cube)
            std = np.nanstd(moisture_cube)
            normalized = (moisture_cube - mean) / (std + 1e-10)
            # Convert z-score to 0-1 range (approximately)
            return np.clip((normalized + 3) / 6, 0, 1)

        if vmax - vmin < 1e-10:
            return np.zeros_like(moisture_cube)

        normalized = (moisture_cube - vmin) / (vmax - vmin)

        return np.clip(normalized, 0, 1)


def apply_exponential_filter(
    timeseries: np.ndarray,
    times: np.ndarray,
    T: float
) -> np.ndarray:
    """
    Apply exponential (SWI-like) filter to a single time series.

    Args:
        timeseries: 1D array of values
        times: 1D array of times (days)
        T: Characteristic time constant

    Returns:
        Filtered time series
    """
    n = len(timeseries)
    filtered = np.full(n, np.nan)

    for t in range(n):
        if np.isnan(timeseries[t]) or np.isnan(times[t]):
            continue

        # Exponential weights for previous observations
        weights = np.exp(-(times[t] - times[:t + 1]) / T)
        valid = ~np.isnan(timeseries[:t + 1]) & ~np.isnan(times[:t + 1])

        w = weights[valid]
        v = timeseries[:t + 1][valid]

        if len(w) > 0:
            filtered[t] = np.sum(v * w) / np.sum(w)

    return filtered


def compute_anomaly(
    moisture_cube: np.ndarray,
    dates: List[datetime],
    climatology_window: int = 30
) -> np.ndarray:
    """
    Compute soil moisture anomaly.

    Anomaly = (SM - climatological_mean) / climatological_std

    Args:
        moisture_cube: Moisture time series
        dates: Acquisition dates
        climatology_window: Days for climatology calculation

    Returns:
        Anomaly time series
    """
    n_dates, n_row, n_col = moisture_cube.shape
    anomaly = np.full_like(moisture_cube, np.nan)

    doy = np.array([d.timetuple().tm_yday if d else np.nan for d in dates])

    for t in range(n_dates):
        if np.isnan(doy[t]):
            continue

        # Find dates within climatology window (same DOY across years)
        doy_diff = np.abs(doy - doy[t])
        doy_diff = np.minimum(doy_diff, 365 - doy_diff)  # Handle year wrap
        climate_mask = doy_diff <= climatology_window

        if np.sum(climate_mask) < 3:
            continue

        climate_data = moisture_cube[climate_mask]
        clim_mean = np.nanmean(climate_data, axis=0)
        clim_std = np.nanstd(climate_data, axis=0)

        clim_std[clim_std < 0.01] = np.nan
        anomaly[t] = (moisture_cube[t] - clim_mean) / clim_std

    return anomaly


def resample_to_dates(
    moisture_cube: np.ndarray,
    source_dates: List[datetime],
    target_dates: List[datetime],
    method: str = 'linear'
) -> np.ndarray:
    """
    Resample moisture data to target dates.

    Args:
        moisture_cube: Source moisture data
        source_dates: Source acquisition dates
        target_dates: Target dates
        method: Interpolation method ('linear', 'nearest')

    Returns:
        Resampled moisture cube
    """
    from scipy import interpolate

    n_target = len(target_dates)
    n_row, n_col = moisture_cube.shape[1:]

    # Convert dates to ordinal
    source_ord = np.array([d.toordinal() if d else np.nan for d in source_dates])
    target_ord = np.array([d.toordinal() for d in target_dates])

    resampled = np.full((n_target, n_row, n_col), np.nan, dtype=np.float32)

    for r in range(n_row):
        for c in range(n_col):
            ts = moisture_cube[:, r, c]
            valid = ~np.isnan(ts) & ~np.isnan(source_ord)

            if np.sum(valid) < 2:
                continue

            try:
                if method == 'nearest':
                    f = interpolate.interp1d(
                        source_ord[valid], ts[valid],
                        kind='nearest', bounds_error=False
                    )
                else:
                    f = interpolate.interp1d(
                        source_ord[valid], ts[valid],
                        kind='linear', bounds_error=False
                    )

                resampled[:, r, c] = f(target_ord)

            except ValueError:
                continue

    return resampled
