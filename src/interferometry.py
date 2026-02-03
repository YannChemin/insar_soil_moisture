"""
Interferometric processing functions for InSAR Soil Moisture retrieval.

This module contains core interferometric operations including:
- Interferogram computation
- Multi-looking
- Coherence estimation
- Closure phase calculation
- Phase linking
"""

import numpy as np
from scipy import ndimage
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


def compute_interferogram(
    slc1: np.ndarray,
    slc2: np.ndarray
) -> np.ndarray:
    """
    Compute complex interferogram between two SLC images.

    The interferogram phase represents the phase difference between
    the two acquisitions.

    Args:
        slc1: First SLC image (complex)
        slc2: Second SLC image (complex)

    Returns:
        Complex interferogram: slc1 * conj(slc2)

    Example:
        >>> slc1 = np.array([[1+1j, 2+2j], [3+3j, 4+4j]])
        >>> slc2 = np.array([[1+0j, 1+1j], [2+1j, 2+2j]])
        >>> ifg = compute_interferogram(slc1, slc2)
    """
    return slc1 * np.conj(slc2)


def multilook(
    data: np.ndarray,
    looks_range: int,
    looks_azimuth: int,
    method: str = 'mean'
) -> np.ndarray:
    """
    Apply multi-looking (spatial averaging) to data.

    Multi-looking reduces speckle noise by averaging neighboring pixels.

    Args:
        data: Input array (can be complex)
        looks_range: Number of looks in range direction
        looks_azimuth: Number of looks in azimuth direction
        method: Averaging method ('mean', 'sum', 'median')

    Returns:
        Multi-looked array

    Note:
        For complex data, this performs coherent averaging.
        For incoherent averaging, apply to magnitude separately.
    """
    if looks_range == 1 and looks_azimuth == 1:
        return data.copy()

    kernel = np.ones((looks_azimuth, looks_range), dtype=np.float32)

    if method == 'sum':
        pass
    elif method == 'median':
        # For median, need different approach
        return _multilook_median(data, looks_range, looks_azimuth)
    else:  # mean
        kernel /= kernel.sum()

    if np.iscomplexobj(data):
        # Separate real and imaginary for convolution
        real_ml = ndimage.convolve(data.real, kernel, mode='constant')
        imag_ml = ndimage.convolve(data.imag, kernel, mode='constant')
        return real_ml + 1j * imag_ml
    else:
        return ndimage.convolve(data, kernel, mode='constant')


def _multilook_median(
    data: np.ndarray,
    looks_range: int,
    looks_azimuth: int
) -> np.ndarray:
    """Multi-looking using median filter."""
    if np.iscomplexobj(data):
        # For complex, compute median of magnitude and circular mean of phase
        mag = np.abs(data)
        phase = np.angle(data)

        mag_ml = ndimage.median_filter(mag, size=(looks_azimuth, looks_range))

        # Circular mean for phase (approximate)
        sin_ml = ndimage.uniform_filter(np.sin(phase), size=(looks_azimuth, looks_range))
        cos_ml = ndimage.uniform_filter(np.cos(phase), size=(looks_azimuth, looks_range))
        phase_ml = np.arctan2(sin_ml, cos_ml)

        return mag_ml * np.exp(1j * phase_ml)
    else:
        return ndimage.median_filter(data, size=(looks_azimuth, looks_range))


def compute_coherence(
    slc1: np.ndarray,
    slc2: np.ndarray,
    looks_range: int,
    looks_azimuth: int
) -> np.ndarray:
    """
    Compute interferometric coherence magnitude.

    Coherence is a measure of the similarity between two SLC images,
    ranging from 0 (no correlation) to 1 (perfect correlation).

    Formula:
        γ = |<s1 * s2*>| / sqrt(<|s1|²> * <|s2|²>)

    Args:
        slc1: First SLC image
        slc2: Second SLC image
        looks_range: Number of looks in range direction
        looks_azimuth: Number of looks in azimuth direction

    Returns:
        Coherence magnitude array (0 to 1)
    """
    # Compute multilooked interferogram
    ifg = compute_interferogram(slc1, slc2)
    ifg_ml = multilook(ifg, looks_range, looks_azimuth)

    # Compute multilooked intensities
    pow1 = multilook(np.abs(slc1) ** 2, looks_range, looks_azimuth)
    pow2 = multilook(np.abs(slc2) ** 2, looks_range, looks_azimuth)

    # Compute coherence
    denom = np.sqrt(pow1 * pow2)
    denom[denom == 0] = np.nan

    coherence = np.abs(ifg_ml) / denom

    return np.clip(coherence, 0, 1)


def compute_closure_phase(
    slc1: np.ndarray,
    slc2: np.ndarray,
    slc3: np.ndarray,
    looks_range: int,
    looks_azimuth: int
) -> np.ndarray:
    """
    Compute closure phase for a triplet of acquisitions.

    The closure phase is defined as:
        φ_closure = arg(⟨ifg12⟩ * ⟨ifg23⟩ * ⟨ifg31⟩)

    For point scatterers, closure phase is always zero. Non-zero closure
    phases indicate distributed scattering or soil moisture variations.

    Args:
        slc1: First SLC image
        slc2: Second SLC image
        slc3: Third SLC image
        looks_range: Number of looks in range direction
        looks_azimuth: Number of looks in azimuth direction

    Returns:
        Closure phase array (radians, -π to π)

    Reference:
        De Zan et al. (2015). Phase inconsistencies and multiple scattering
        in SAR interferometry. IEEE TGRS.
    """
    # Compute multilooked interferograms
    ifg12 = multilook(compute_interferogram(slc1, slc2), looks_range, looks_azimuth)
    ifg23 = multilook(compute_interferogram(slc2, slc3), looks_range, looks_azimuth)
    ifg31 = multilook(compute_interferogram(slc3, slc1), looks_range, looks_azimuth)

    # Closure phase is the phase of the triple product
    closure = ifg12 * ifg23 * ifg31

    return np.angle(closure)


def phase_linking(
    slc_stack: np.ndarray,
    looks_range: int,
    looks_azimuth: int,
    method: str = 'evd'
) -> np.ndarray:
    """
    Perform phase linking to estimate consistent phase time series.

    Phase linking finds the best phase estimate for each acquisition
    by considering all possible interferograms jointly.

    Args:
        slc_stack: Stack of SLC images (N, rows, cols)
        looks_range: Number of looks in range direction
        looks_azimuth: Number of looks in azimuth direction
        method: Phase linking method ('evd', 'mle')

    Returns:
        Linked phases (N, rows, cols) in radians

    Reference:
        Ansari et al. (2018). Efficient phase estimation for interferogram stacks.
        IEEE TGRS.
    """
    n_images, rows, cols = slc_stack.shape

    # Compute coherence matrix
    coh_matrix = np.zeros((n_images, n_images, rows, cols), dtype=np.complex64)

    for i in range(n_images):
        for j in range(i, n_images):
            if i == j:
                coh_matrix[i, j] = 1.0
            else:
                ifg = compute_interferogram(slc_stack[i], slc_stack[j])
                ifg_ml = multilook(ifg, looks_range, looks_azimuth)

                pow_i = multilook(np.abs(slc_stack[i]) ** 2, looks_range, looks_azimuth)
                pow_j = multilook(np.abs(slc_stack[j]) ** 2, looks_range, looks_azimuth)

                denom = np.sqrt(pow_i * pow_j)
                denom[denom == 0] = 1

                coh_matrix[i, j] = ifg_ml / denom
                coh_matrix[j, i] = np.conj(coh_matrix[i, j])

    # Phase linking using eigenvalue decomposition
    linked_phases = np.zeros((n_images, rows, cols))

    if method == 'evd':
        linked_phases = _phase_linking_evd(coh_matrix)
    else:
        # Default to simple sequential integration
        linked_phases = _phase_linking_sequential(slc_stack, looks_range, looks_azimuth)

    return linked_phases


def _phase_linking_evd(coh_matrix: np.ndarray) -> np.ndarray:
    """Phase linking using eigenvalue decomposition."""
    n_images = coh_matrix.shape[0]
    rows, cols = coh_matrix.shape[2:]

    linked_phases = np.zeros((n_images, rows, cols))

    # Process each pixel
    for r in range(rows):
        for c in range(cols):
            C = coh_matrix[:, :, r, c]

            # Eigenvalue decomposition
            eigenvalues, eigenvectors = np.linalg.eigh(C)

            # Principal eigenvector corresponds to linked phases
            principal = eigenvectors[:, -1]
            linked_phases[:, r, c] = np.angle(principal)

            # Reference to first acquisition
            linked_phases[:, r, c] -= linked_phases[0, r, c]

    return linked_phases


def _phase_linking_sequential(
    slc_stack: np.ndarray,
    looks_range: int,
    looks_azimuth: int
) -> np.ndarray:
    """Simple sequential phase integration (baseline approach)."""
    n_images, rows, cols = slc_stack.shape
    linked_phases = np.zeros((n_images, rows, cols))

    for i in range(1, n_images):
        ifg = compute_interferogram(slc_stack[i], slc_stack[i - 1])
        ifg_ml = multilook(ifg, looks_range, looks_azimuth)
        linked_phases[i] = linked_phases[i - 1] + np.angle(ifg_ml)

    return linked_phases


def compute_covariance_matrix(
    slc_window: np.ndarray
) -> np.ndarray:
    """
    Compute sample covariance matrix from SLC data.

    Args:
        slc_window: SLC data for window (N_images, N_pixels)

    Returns:
        Covariance matrix (N_images, N_images)
    """
    n_images, n_pixels = slc_window.shape

    # Normalize by pixel power
    normalized = slc_window.copy()
    for i in range(n_images):
        power = np.sqrt(np.mean(np.abs(slc_window[i]) ** 2))
        if power > 0:
            normalized[i] /= power

    # Sample covariance
    C = (normalized @ normalized.conj().T) / n_pixels

    return C


def generate_synthetic_reference(
    slc_triplet: np.ndarray,
    target_phases: np.ndarray
) -> np.ndarray:
    """
    Generate synthetic reference image from SLC triplet.

    Creates a reference image that, when interfered with the original
    acquisitions, produces the target phase history.

    Args:
        slc_triplet: Three SLC images (3, n_pixels)
        target_phases: Target phases (3,)

    Returns:
        Synthetic reference image (n_pixels,)

    Formula:
        z = y * C_y^(-1) * ξ^H
        where ξ = [exp(jφ_1), exp(jφ_2), exp(jφ_3)]
    """
    # Compute covariance matrix
    C_y = compute_covariance_matrix(slc_triplet)

    # Target phase vector
    xi = np.exp(1j * target_phases)

    # Invert covariance (with regularization)
    try:
        C_y_inv = np.linalg.inv(C_y)
    except np.linalg.LinAlgError:
        # Add regularization
        C_y_reg = C_y + 0.01 * np.eye(3)
        C_y_inv = np.linalg.inv(C_y_reg)

    # Generate reference
    z = slc_triplet.T @ C_y_inv @ xi.conj()

    return z


def unwrap_phase(
    phase: np.ndarray,
    method: str = 'goldstein'
) -> np.ndarray:
    """
    Unwrap 2D phase array.

    Phase unwrapping resolves the 2π ambiguity in interferometric phase.

    Args:
        phase: Wrapped phase array (radians)
        method: Unwrapping method ('goldstein', 'snaphu')

    Returns:
        Unwrapped phase array

    Note:
        For full SNAPHU unwrapping, use the snaphu package.
        This provides a basic implementation.
    """
    if method == 'snaphu':
        try:
            import snaphu
            return snaphu.unwrap(phase)
        except ImportError:
            logger.warning("SNAPHU not available, using basic unwrapping")

    # Basic Goldstein-style unwrapping (simplified)
    return _goldstein_unwrap(phase)


def _goldstein_unwrap(phase: np.ndarray) -> np.ndarray:
    """Basic phase unwrapping using path integration."""
    rows, cols = phase.shape
    unwrapped = np.zeros_like(phase)

    # Reference at center
    ref_r, ref_c = rows // 2, cols // 2

    # Simple row-by-row unwrapping from reference
    for r in range(rows):
        for c in range(cols):
            if r == ref_r and c == ref_c:
                unwrapped[r, c] = phase[r, c]
                continue

            # Find path to reference
            if c > 0:
                diff = phase[r, c] - phase[r, c - 1]
                unwrapped[r, c] = unwrapped[r, c - 1] + _wrap(diff)
            elif r > 0:
                diff = phase[r, c] - phase[r - 1, c]
                unwrapped[r, c] = unwrapped[r - 1, c] + _wrap(diff)
            else:
                unwrapped[r, c] = phase[r, c]

    return unwrapped


def _wrap(phase: float) -> float:
    """Wrap phase to [-π, π]."""
    return np.arctan2(np.sin(phase), np.cos(phase))


def estimate_coherence_bias(
    coherence: float,
    n_looks: int
) -> float:
    """
    Estimate coherence bias due to finite number of looks.

    Args:
        coherence: Measured coherence
        n_looks: Number of independent looks

    Returns:
        Bias-corrected coherence estimate

    Reference:
        Touzi & Lopes (1996). Statistics of the Stokes parameters and of
        the complex coherence parameters in one-look and multilook
        speckle fields.
    """
    # Approximate bias correction
    if coherence >= 1:
        return 1.0

    bias = (1 - coherence ** 2) / (2 * n_looks)

    corrected = np.sqrt(max(0, coherence ** 2 - bias))

    return min(corrected, 1.0)
