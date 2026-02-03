"""
Main processor for InSAR-based Soil Moisture retrieval.

This module implements the core algorithm from De Zan et al. (2026).
"""

import numpy as np
from typing import Tuple, Optional, List, Dict, Any
from dataclasses import dataclass
import logging
from datetime import datetime

from .config import Config
from .interferometry import (
    compute_interferogram,
    multilook,
    compute_coherence,
    compute_closure_phase,
    compute_covariance_matrix,
    generate_synthetic_reference,
)

logger = logging.getLogger(__name__)


@dataclass
class ProcessingResult:
    """Container for processing results."""
    moisture_cube: np.ndarray
    coherence_map: np.ndarray
    quality_flags: np.ndarray
    metadata: Dict[str, Any]

    @property
    def mean_coherence(self) -> float:
        """Mean coherence over all pixels."""
        return float(np.nanmean(self.coherence_map))

    @property
    def valid_fraction(self) -> float:
        """Fraction of valid (non-NaN) pixels."""
        return 1 - np.mean(np.isnan(self.moisture_cube[0]))


class InSARSoilMoistureProcessor:
    """
    InSAR-based soil moisture retrieval processor.

    Implements the algorithm from De Zan et al. (2026) for retrieving
    high-resolution soil moisture from SAR interferometric closure phases.

    Attributes:
        config: Processing configuration

    Example:
        >>> config = Config()
        >>> processor = InSARSoilMoistureProcessor(config)
        >>> result = processor.process_stack(stack, metadata)
    """

    def __init__(self, config: Optional[Config] = None):
        """
        Initialize processor.

        Args:
            config: Processing configuration (uses defaults if None)
        """
        self.config = config or Config()
        logger.info(f"Initialized processor with config:\n{self.config}")

    def process_stack(
        self,
        stack: np.ndarray,
        metadata: Dict[str, Any]
    ) -> ProcessingResult:
        """
        Process entire SLC stack to retrieve soil moisture.

        Args:
            stack: Complex SLC stack (n_images, rows, cols)
            metadata: Dictionary with geotransform, projection, dates

        Returns:
            ProcessingResult with moisture cube and quality metrics
        """
        n_images, rows, cols = stack.shape
        lr = self.config.WINDOW_RANGE
        la = self.config.WINDOW_AZIMUTH

        n_win_row = rows // la
        n_win_col = cols // lr

        logger.info(f"Processing {n_images} images over {n_win_row}x{n_win_col} windows")

        # Initialize outputs
        moisture_cube = np.full((n_images, n_win_row, n_win_col), np.nan, dtype=np.float32)
        coherence_map = np.zeros((n_win_row, n_win_col), dtype=np.float32)
        quality_flags = np.zeros((n_win_row, n_win_col), dtype=np.uint8)

        # Process each window
        processed = 0
        total_windows = n_win_row * n_win_col

        for wr in range(n_win_row):
            for wc in range(n_win_col):
                window_row = wr * la
                window_col = wc * lr

                result = self._process_window(stack, window_row, window_col)

                if result is not None:
                    moisture, coh, flag = result
                    moisture_cube[:, wr, wc] = moisture
                    coherence_map[wr, wc] = coh
                    quality_flags[wr, wc] = flag
                    processed += 1

            # Progress logging
            if (wr + 1) % max(1, n_win_row // 10) == 0:
                pct = 100 * (wr + 1) / n_win_row
                logger.info(f"Progress: {pct:.0f}% ({processed}/{(wr+1)*n_win_col} valid windows)")

        logger.info(f"Processed {processed}/{total_windows} windows "
                   f"({100*processed/total_windows:.1f}% valid)")

        return ProcessingResult(
            moisture_cube=moisture_cube,
            coherence_map=coherence_map,
            quality_flags=quality_flags,
            metadata={
                'n_images': n_images,
                'n_windows': (n_win_row, n_win_col),
                'valid_windows': processed,
                'dates': metadata.get('dates', []),
            }
        )

    def _process_window(
        self,
        stack: np.ndarray,
        window_row: int,
        window_col: int
    ) -> Optional[Tuple[np.ndarray, float, int]]:
        """
        Process a single multi-look window.

        Args:
            stack: Full SLC stack
            window_row: Starting row of window
            window_col: Starting column of window

        Returns:
            Tuple of (moisture_timeseries, mean_coherence, quality_flag)
            or None if processing failed
        """
        lr = self.config.WINDOW_RANGE
        la = self.config.WINDOW_AZIMUTH

        # Extract window data
        r0, r1 = window_row, window_row + la
        c0, c1 = window_col, window_col + lr

        if r1 > stack.shape[1] or c1 > stack.shape[2]:
            return None

        local_stack = stack[:, r0:r1, c0:c1]

        # Step 1: Find best triplet
        triplet = self._find_best_triplet(local_stack)
        if triplet is None:
            return None

        # Step 2: Invert moisture order
        moisture_levels = self._invert_moisture_order(local_stack, triplet)

        # Steps 3-4: Compute phase histories and references
        phi_a, phi_b = self._compute_phase_histories(local_stack, triplet, moisture_levels)
        z_a, z_b = self._generate_references(local_stack, triplet, phi_a, phi_b)

        # Step 5: Retrieve moisture time series
        moisture = self._retrieve_moisture_timeseries(local_stack, z_a, z_b)

        # Compute mean coherence
        mean_coh = self._compute_mean_coherence(local_stack)

        # Quality flag
        flag = 1  # Valid

        return moisture, mean_coh, flag

    def _find_best_triplet(
        self,
        local_stack: np.ndarray
    ) -> Optional[Tuple[int, int, int]]:
        """
        Step 1: Find triplet with largest closure phase.

        Subject to: product of three coherence magnitudes > threshold
        """
        n_images = local_stack.shape[0]

        if n_images < 3:
            return None

        best_triplet = None
        max_closure = 0

        # Search consecutive triplets
        for i in range(n_images - 2):
            j, k = i + 1, i + 2

            slc_i = local_stack[i]
            slc_j = local_stack[j]
            slc_k = local_stack[k]

            # Compute coherences
            coh_ij = self._mean_coherence(slc_i, slc_j)
            coh_jk = self._mean_coherence(slc_j, slc_k)
            coh_ki = self._mean_coherence(slc_k, slc_i)

            coh_product = coh_ij * coh_jk * coh_ki

            if coh_product < self.config.TRIPLET_COHERENCE_THRESHOLD:
                continue

            # Compute closure phase
            closure = self._mean_closure_phase(slc_i, slc_j, slc_k)
            abs_closure = np.abs(closure)

            if abs_closure > max_closure:
                max_closure = abs_closure
                best_triplet = (i, j, k)

        return best_triplet

    def _mean_coherence(self, slc1: np.ndarray, slc2: np.ndarray) -> float:
        """Compute mean coherence over window."""
        ifg = slc1 * np.conj(slc2)
        ifg_mean = np.nanmean(ifg)

        pow1 = np.nanmean(np.abs(slc1) ** 2)
        pow2 = np.nanmean(np.abs(slc2) ** 2)

        denom = np.sqrt(pow1 * pow2)
        if denom == 0:
            return 0

        return float(np.abs(ifg_mean) / denom)

    def _mean_closure_phase(
        self,
        slc1: np.ndarray,
        slc2: np.ndarray,
        slc3: np.ndarray
    ) -> float:
        """Compute mean closure phase over window."""
        ifg12 = np.nanmean(slc1 * np.conj(slc2))
        ifg23 = np.nanmean(slc2 * np.conj(slc3))
        ifg31 = np.nanmean(slc3 * np.conj(slc1))

        return float(np.angle(ifg12 * ifg23 * ifg31))

    def _invert_moisture_order(
        self,
        local_stack: np.ndarray,
        triplet: Tuple[int, int, int]
    ) -> Tuple[float, float, float]:
        """
        Step 2: Assign nominal moisture levels (0.1, 0.2, 0.3) to triplet.

        Uses coherence to identify largest moisture difference,
        then closure phase sign to determine ordering.
        """
        i, j, k = triplet

        slc_i = local_stack[i]
        slc_j = local_stack[j]
        slc_k = local_stack[k]

        # Compute coherences
        coh_ij = self._mean_coherence(slc_i, slc_j)
        coh_jk = self._mean_coherence(slc_j, slc_k)
        coh_ik = self._mean_coherence(slc_i, slc_k)

        # Find pair with lowest coherence (largest moisture difference)
        coherences = {'ij': coh_ij, 'jk': coh_jk, 'ik': coh_ik}
        min_coh_pair = min(coherences, key=coherences.get)

        # Compute closure phase sign
        closure = self._mean_closure_phase(slc_i, slc_j, slc_k)

        # Assign moisture levels based on coherence and closure phase
        # Positive closure = increasing moisture in circular order (De Zan et al. 2012)
        if min_coh_pair == 'ik':  # i and k most different
            if closure > 0:
                return (0.1, 0.2, 0.3)
            else:
                return (0.3, 0.2, 0.1)
        elif min_coh_pair == 'ij':  # i and j most different
            if closure > 0:
                return (0.2, 0.3, 0.1)
            else:
                return (0.2, 0.1, 0.3)
        else:  # jk most different
            if closure > 0:
                return (0.3, 0.1, 0.2)
            else:
                return (0.1, 0.3, 0.2)

    def _compute_phase_histories(
        self,
        local_stack: np.ndarray,
        triplet: Tuple[int, int, int],
        moisture_levels: Tuple[float, float, float]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Steps 3: Determine two phase histories.

        φ_a: Phase linking result
        φ_b: φ_a + β * moisture_level
        """
        i, j, k = triplet
        m_i, m_j, m_k = moisture_levels

        slc_i = local_stack[i]
        slc_j = local_stack[j]
        slc_k = local_stack[k]

        # Phase linking for φ_a (simplified: phases relative to first image)
        phi_a_i = 0.0
        phi_a_j = np.angle(np.nanmean(slc_j * np.conj(slc_i)))
        phi_a_k = np.angle(np.nanmean(slc_k * np.conj(slc_i)))

        # Compute β from closure phase magnitude
        closure = self._mean_closure_phase(slc_i, slc_j, slc_k)
        beta = np.cbrt(np.abs(closure))  # Cube root as per De Zan et al.

        # φ_b = φ_a + β * m
        phi_b_i = phi_a_i + beta * m_i
        phi_b_j = phi_a_j + beta * m_j
        phi_b_k = phi_a_k + beta * m_k

        phi_a = np.array([phi_a_i, phi_a_j, phi_a_k])
        phi_b = np.array([phi_b_i, phi_b_j, phi_b_k])

        return phi_a, phi_b

    def _generate_references(
        self,
        local_stack: np.ndarray,
        triplet: Tuple[int, int, int],
        phi_a: np.ndarray,
        phi_b: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Step 4: Generate synthetic reference images z_a and z_b.

        z = y * C_y^(-1) * ξ^H
        """
        i, j, k = triplet

        # Extract and flatten the three SLCs
        slc_triplet = np.array([
            local_stack[i].flatten(),
            local_stack[j].flatten(),
            local_stack[k].flatten()
        ])

        # Generate references
        z_a = generate_synthetic_reference(slc_triplet, phi_a)
        z_b = generate_synthetic_reference(slc_triplet, phi_b)

        return z_a, z_b

    def _retrieve_moisture_timeseries(
        self,
        local_stack: np.ndarray,
        z_a: np.ndarray,
        z_b: np.ndarray
    ) -> np.ndarray:
        """
        Step 5: Retrieve moisture time series from phase differences.

        m̂(t) ∝ ∠(z_a^H * y(t)) - ∠(z_b^H * y(t))
        """
        n_images = local_stack.shape[0]
        moisture = np.zeros(n_images, dtype=np.float32)

        for t in range(n_images):
            y_t = local_stack[t].flatten()

            # Interferometric phases with references
            phi_ta = np.angle(np.sum(z_a.conj() * y_t))
            phi_tb = np.angle(np.sum(z_b.conj() * y_t))

            moisture[t] = phi_ta - phi_tb

        return moisture

    def _compute_mean_coherence(self, local_stack: np.ndarray) -> float:
        """Compute mean temporal coherence for quality assessment."""
        n_images = local_stack.shape[0]

        coherences = []
        for i in range(min(n_images - 1, 10)):
            coh = self._mean_coherence(local_stack[i], local_stack[i + 1])
            coherences.append(coh)

        return float(np.mean(coherences)) if coherences else 0.0


class TiledProcessor:
    """
    Memory-efficient processor using tiled processing.

    For large datasets that don't fit in memory.
    """

    def __init__(
        self,
        config: Optional[Config] = None,
        tile_size: Tuple[int, int] = (1000, 1000),
        overlap: int = 100
    ):
        """
        Initialize tiled processor.

        Args:
            config: Processing configuration
            tile_size: Tile size (rows, cols)
            overlap: Overlap between tiles in pixels
        """
        self.config = config or Config()
        self.tile_size = tile_size
        self.overlap = overlap
        self.processor = InSARSoilMoistureProcessor(config)

    def process_tiled(
        self,
        slc_paths: List[str],
        output_dir: str
    ) -> str:
        """
        Process SLC stack using tiled approach.

        Args:
            slc_paths: List of SLC file paths
            output_dir: Output directory

        Returns:
            Path to output file
        """
        from .io_utils import read_slc_stack, write_geotiff, get_raster_info
        import os

        # Get image dimensions from first file
        info = get_raster_info(slc_paths[0])
        total_rows = info['height']
        total_cols = info['width']

        tile_rows, tile_cols = self.tile_size

        n_tiles_row = (total_rows + tile_rows - 1) // tile_rows
        n_tiles_col = (total_cols + tile_cols - 1) // tile_cols

        logger.info(f"Processing {n_tiles_row}x{n_tiles_col} tiles")

        # Process each tile
        for tr in range(n_tiles_row):
            for tc in range(n_tiles_col):
                r0 = max(0, tr * tile_rows - self.overlap)
                r1 = min(total_rows, (tr + 1) * tile_rows + self.overlap)
                c0 = max(0, tc * tile_cols - self.overlap)
                c1 = min(total_cols, (tc + 1) * tile_cols + self.overlap)

                logger.info(f"Processing tile ({tr}, {tc}): rows {r0}-{r1}, cols {c0}-{c1}")

                # Read tile data
                # ... (implementation for reading subsets)

                # Process tile
                # ... (implementation for tile processing)

        # Merge tiles
        # ... (implementation for merging)

        output_path = os.path.join(output_dir, 'insar_soil_moisture.tif')
        return output_path
