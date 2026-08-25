"""
Tests for the files module (Filenames class).
"""

import pytest
from emit_ghg.files import Filenames


def test_filenames_creation():
    """Test that Filenames class creates correct file paths."""
    base_path = "/path/to/output/emit20230101t120000"
    files = Filenames(base_path)

    # Check generated paths contain the base
    assert base_path in files.target_file
    assert base_path in files.mf_file
    assert base_path in files.mf_ort_file

    # Check that all expected attributes exist
    assert hasattr(files, 'target_file')
    assert hasattr(files, 'mf_file')
    assert hasattr(files, 'mf_uncert_file')
    assert hasattr(files, 'mf_sens_file')
    assert hasattr(files, 'flare_file')
    assert hasattr(files, 'mf_ort_file')
    assert hasattr(files, 'mf_ort_cog')
    assert hasattr(files, 'mf_ort_cog_d1')
    assert hasattr(files, 'mf_ort_cog_d2')
    assert hasattr(files, 'sens_ort_file')
    assert hasattr(files, 'uncert_ort_file')


def test_filenames_extensions():
    """Test that Filenames generates correct file extensions."""
    base_path = "/test/emit20230101t120000"
    files = Filenames(base_path)

    # Check that generated files have appropriate extensions or patterns
    assert "target" in files.target_file
    assert "mf" in files.mf_file
    assert "ort" in files.mf_ort_file
