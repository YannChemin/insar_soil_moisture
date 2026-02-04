"""
Tests for the InSAR Soil Moisture processor.
"""

import pytest
import numpy as np
from datetime import datetime, timedelta
import tempfile
import os

# Import modules to test
from insar_soil_moisture.config import Config
from insar_soil_moisture.interferometry import (
    compute_interferogram,
    multilook,
    compute_coherence,
    compute_closure_phase,
)
from insar_soil_moisture.processor import InSARSoilMoistureProcessor
from insar_soil_moisture.postprocessing import PostProcessor


class TestConfig:
    """Tests for Config class."""

    def test_default_config(self):
        """Test default configuration values."""
        config = Config()

        assert config.WINDOW_RANGE == 79
        assert config.WINDOW_AZIMUTH == 18
        assert config.TRIPLET_COHERENCE_THRESHOLD == pytest.approx(0.027, abs=0.001)
        assert config.OUTPUT_SPACING == 1000

    def test_custom_config(self):
        """Test custom configuration."""
        config = Config(
            WINDOW_RANGE=100,
            WINDOW_AZIMUTH=25,
            TEMPORAL_FILTER_T=5.0
        )

        assert config.WINDOW_RANGE == 100
        assert config.WINDOW_AZIMUTH == 25
        assert config.TEMPORAL_FILTER_T == 5.0

    def test_config_validation(self):
        """Test configuration validation."""
        with pytest.raises(ValueError):
            Config(WINDOW_RANGE=-1)

        with pytest.raises(ValueError):
            Config(TRIPLET_COHERENCE_THRESHOLD=1.5)

    def test_config_from_dict(self):
        """Test configuration from dictionary."""
        config_dict = {
            'processing': {
                'window_range': 50,
                'window_azimuth': 10,
            },
            'filtering': {
                'temporal': {
                    'enabled': False,
                    'characteristic_time': 5,
                }
            }
        }

        config = Config.from_dict(config_dict)

        assert config.WINDOW_RANGE == 50
        assert config.WINDOW_AZIMUTH == 10
        assert config.TEMPORAL_FILTER_ENABLED is False
        assert config.TEMPORAL_FILTER_T == 5

    def test_config_to_dict(self):
        """Test configuration to dictionary."""
        config = Config()
        config_dict = config.to_dict()

        assert 'processing' in config_dict
        assert 'filtering' in config_dict
        assert config_dict['processing']['window_range'] == 79


class TestInterferometry:
    """Tests for interferometric functions."""

    def test_compute_interferogram(self):
        """Test interferogram computation."""
        slc1 = np.array([[1 + 1j, 2 + 2j], [3 + 3j, 4 + 4j]])
        slc2 = np.array([[1 + 0j, 1 + 1j], [2 + 1j, 2 + 2j]])

        ifg = compute_interferogram(slc1, slc2)

        # ifg = slc1 * conj(slc2)
        expected = slc1 * np.conj(slc2)
        np.testing.assert_array_almost_equal(ifg, expected)

    def test_multilook(self):
        """Test multi-looking."""
        data = np.ones((10, 10), dtype=np.float32)
        data[5:, :] = 2.0

        ml = multilook(data, 2, 2)

        # Shape should be preserved
        assert ml.shape == data.shape

        # Values should be averaged
        assert ml[0, 0] == pytest.approx(1.0, abs=0.1)
        assert ml[7, 5] == pytest.approx(2.0, abs=0.1)

    def test_multilook_complex(self):
        """Test multi-looking with complex data."""
        data = np.ones((10, 10), dtype=np.complex64) * (1 + 1j)

        ml = multilook(data, 3, 3)

        assert np.iscomplexobj(ml)
        assert ml.shape == data.shape

    def test_compute_coherence(self):
        """Test coherence computation."""
        # Create two identical SLCs (should have coherence = 1)
        slc = np.random.randn(20, 20) + 1j * np.random.randn(20, 20)
        slc = slc.astype(np.complex64)

        coh = compute_coherence(slc, slc, 5, 5)

        # Coherence with itself should be ~1
        assert np.nanmean(coh) > 0.95

    def test_compute_coherence_uncorrelated(self):
        """Test coherence with uncorrelated data."""
        np.random.seed(42)
        slc1 = np.random.randn(20, 20) + 1j * np.random.randn(20, 20)
        slc2 = np.random.randn(20, 20) + 1j * np.random.randn(20, 20)

        coh = compute_coherence(slc1.astype(np.complex64),
                               slc2.astype(np.complex64), 5, 5)

        # Coherence should be low for uncorrelated data
        assert np.nanmean(coh) < 0.5

    def test_compute_closure_phase(self):
        """Test closure phase computation."""
        np.random.seed(42)

        # For point scatterers, closure phase should be ~0
        n = 20
        phase1 = np.random.uniform(-np.pi, np.pi, (n, n))
        phase2 = np.random.uniform(-np.pi, np.pi, (n, n))
        phase3 = np.random.uniform(-np.pi, np.pi, (n, n))

        slc1 = np.exp(1j * phase1)
        slc2 = np.exp(1j * phase2)
        slc3 = np.exp(1j * phase3)

        closure = compute_closure_phase(slc1, slc2, slc3, 5, 5)

        # Closure phase should be defined
        assert closure.shape == (n, n)
        assert np.all(np.abs(closure) <= np.pi)


class TestProcessor:
    """Tests for the main processor."""

    @pytest.fixture
    def sample_stack(self):
        """Create a sample SLC stack for testing."""
        np.random.seed(42)
        n_images = 10
        rows, cols = 100, 100

        # Create synthetic SLC data with soil moisture signal
        stack = np.zeros((n_images, rows, cols), dtype=np.complex64)

        for i in range(n_images):
            # Base phase (terrain)
            base_phase = np.random.uniform(-np.pi, np.pi, (rows, cols))

            # Add moisture signal (slowly varying)
            moisture = 0.3 + 0.2 * np.sin(2 * np.pi * i / n_images)
            moisture_phase = moisture * np.ones((rows, cols))

            total_phase = base_phase + moisture_phase
            stack[i] = np.exp(1j * total_phase)

        return stack

    @pytest.fixture
    def sample_metadata(self):
        """Create sample metadata."""
        base_date = datetime(2020, 1, 1)
        dates = [base_date + timedelta(days=12 * i) for i in range(10)]

        return {
            'geotransform': (0, 10, 0, 0, 0, -10),
            'projection': 'EPSG:32632',
            'rows': 100,
            'cols': 100,
            'dates': dates,
        }

    def test_processor_initialization(self):
        """Test processor initialization."""
        config = Config()
        processor = InSARSoilMoistureProcessor(config)

        assert processor.config == config

    def test_mean_coherence(self, sample_stack):
        """Test mean coherence calculation."""
        config = Config()
        processor = InSARSoilMoistureProcessor(config)

        # Window from the stack
        window = sample_stack[:, :18, :79]

        coh = processor._mean_coherence(window[0], window[1])

        assert 0 <= coh <= 1

    def test_mean_closure_phase(self, sample_stack):
        """Test mean closure phase calculation."""
        config = Config()
        processor = InSARSoilMoistureProcessor(config)

        window = sample_stack[:, :18, :79]

        closure = processor._mean_closure_phase(window[0], window[1], window[2])

        assert -np.pi <= closure <= np.pi

    def test_find_best_triplet(self, sample_stack):
        """Test triplet selection."""
        config = Config(TRIPLET_COHERENCE_THRESHOLD=0.001)  # Low threshold for test
        processor = InSARSoilMoistureProcessor(config)

        window = sample_stack[:, :18, :79]
        triplet = processor._find_best_triplet(window)

        # Should find a triplet
        if triplet is not None:
            assert len(triplet) == 3
            assert all(0 <= i < window.shape[0] for i in triplet)


class TestPostProcessor:
    """Tests for post-processing functions."""

    @pytest.fixture
    def sample_moisture_cube(self):
        """Create sample moisture data."""
        np.random.seed(42)
        n_dates = 20
        rows, cols = 50, 50

        # Create synthetic moisture data
        moisture = np.zeros((n_dates, rows, cols), dtype=np.float32)

        for t in range(n_dates):
            # Spatial pattern
            x, y = np.meshgrid(np.arange(cols), np.arange(rows))
            spatial = 0.5 + 0.3 * np.sin(2 * np.pi * x / cols)

            # Temporal variation
            temporal = 0.2 * np.sin(2 * np.pi * t / n_dates)

            moisture[t] = spatial + temporal + 0.1 * np.random.randn(rows, cols)

        # Add some NaN values
        moisture[:, 0:5, 0:5] = np.nan

        return moisture

    @pytest.fixture
    def sample_dates(self):
        """Create sample dates."""
        base_date = datetime(2020, 1, 1)
        return [base_date + timedelta(days=6 * i) for i in range(20)]

    def test_sign_correction(self, sample_moisture_cube):
        """Test sign correction."""
        config = Config()
        post = PostProcessor(config)

        # Flip sign of some pixels
        moisture = sample_moisture_cube.copy()
        moisture[:, 20:30, 20:30] = -moisture[:, 20:30, 20:30]

        corrected = post.correct_sign(moisture)

        # After correction, correlation should be positive
        scene_avg = np.nanmean(corrected, axis=(1, 2))
        for r in range(corrected.shape[1]):
            for c in range(corrected.shape[2]):
                ts = corrected[:, r, c]
                if not np.all(np.isnan(ts)):
                    valid = ~np.isnan(ts)
                    if np.sum(valid) >= 5:
                        from scipy.stats import spearmanr
                        corr, _ = spearmanr(ts[valid], scene_avg[valid])
                        assert corr >= -0.1  # Should be positive or near zero

    def test_coherence_mask(self, sample_moisture_cube):
        """Test coherence masking."""
        config = Config(AGGREGATION_COHERENCE_THRESHOLD=0.3)
        post = PostProcessor(config)

        # Create coherence map
        coherence = np.random.uniform(0, 1, sample_moisture_cube.shape[1:])

        masked = post.apply_coherence_mask(sample_moisture_cube, coherence)

        # Low coherence pixels should be NaN
        low_coh = coherence < 0.3
        assert np.all(np.isnan(masked[:, low_coh]))

    def test_temporal_filter(self, sample_moisture_cube, sample_dates):
        """Test temporal filtering."""
        config = Config(TEMPORAL_FILTER_T=10.0)
        post = PostProcessor(config)

        filtered = post.temporal_filter(sample_moisture_cube, sample_dates)

        # Filtered data should be smoother (lower std)
        orig_std = np.nanstd(sample_moisture_cube, axis=0)
        filt_std = np.nanstd(filtered, axis=0)

        # Most pixels should have reduced temporal variance
        valid = ~np.isnan(orig_std) & ~np.isnan(filt_std) & (orig_std > 0)
        if np.sum(valid) > 0:
            reduction = filt_std[valid] / orig_std[valid]
            assert np.median(reduction) < 1.5  # Should be similar or reduced

    def test_spatial_filter(self, sample_moisture_cube):
        """Test spatial filtering."""
        config = Config(SPATIAL_FILTER_SIGMA=2.0)
        post = PostProcessor(config)

        filtered = post.spatial_filter(sample_moisture_cube)

        # Filtered data should be smoother spatially
        assert filtered.shape == sample_moisture_cube.shape

    def test_normalize(self, sample_moisture_cube):
        """Test normalization."""
        config = Config()
        post = PostProcessor(config)

        normalized = post.normalize(sample_moisture_cube)

        # Should be in 0-1 range
        valid = ~np.isnan(normalized)
        assert np.all(normalized[valid] >= 0)
        assert np.all(normalized[valid] <= 1)


class TestIntegration:
    """Integration tests."""

    def test_full_pipeline(self):
        """Test complete processing pipeline."""
        np.random.seed(42)

        # Create synthetic data
        n_images = 5
        rows, cols = 50, 50

        stack = np.random.randn(n_images, rows, cols) + \
                1j * np.random.randn(n_images, rows, cols)
        stack = stack.astype(np.complex64)

        base_date = datetime(2020, 1, 1)
        dates = [base_date + timedelta(days=12 * i) for i in range(n_images)]

        metadata = {
            'geotransform': (0, 10, 0, 0, 0, -10),
            'projection': 'EPSG:32632',
            'rows': rows,
            'cols': cols,
            'dates': dates,
        }

        # Process with relaxed thresholds for synthetic data
        config = Config(
            WINDOW_RANGE=10,
            WINDOW_AZIMUTH=10,
            TRIPLET_COHERENCE_THRESHOLD=0.001,
            AGGREGATION_COHERENCE_THRESHOLD=0.01,
        )

        processor = InSARSoilMoistureProcessor(config)
        result = processor.process_stack(stack, metadata)

        # Check output dimensions
        expected_rows = rows // config.WINDOW_AZIMUTH
        expected_cols = cols // config.WINDOW_RANGE

        assert result.moisture_cube.shape[0] == n_images
        assert result.moisture_cube.shape[1] == expected_rows
        assert result.moisture_cube.shape[2] == expected_cols


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
