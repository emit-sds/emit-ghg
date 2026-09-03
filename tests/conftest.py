"""
Pytest configuration and fixtures for EMIT GHG tests.
"""

import pytest
import numpy as np
import os


@pytest.fixture
def sample_radiance_data():
    """Generate synthetic radiance data for testing."""
    # Create a small sample radiance cube (10x10x100 bands)
    lines, samples, bands = 10, 10, 100
    radiance = np.random.rand(lines, samples, bands).astype(np.float32) * 1000
    return radiance


@pytest.fixture
def sample_wavelengths():
    """Generate sample wavelength array."""
    return np.linspace(400, 2500, 100)


@pytest.fixture
def sample_target():
    """Generate a sample target signature."""
    bands = 100
    target = np.random.rand(bands).astype(np.float32) * 0.1
    return target


@pytest.fixture
def temp_output_dir(tmp_path):
    """Create a temporary output directory."""
    output_dir = tmp_path / "emit_ghg_test_output"
    output_dir.mkdir()
    return str(output_dir)


@pytest.fixture
def noise_file_path():
    """Return path to the instrument noise parameters file."""
    import emit_ghg
    package_dir = os.path.dirname(emit_ghg.__file__)
    noise_file = os.path.join(package_dir, 'data', 'instrument_noise_parameters', 'emit_noise.txt')
    if os.path.exists(noise_file):
        return noise_file
    return None
