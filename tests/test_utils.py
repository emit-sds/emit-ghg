"""
Tests for utility functions.
"""

import pytest
import os
from emit_ghg.utils import envi_header


def test_envi_header_img_extension():
    """Test envi_header with .img extension."""
    input_path = "/path/to/file.img"
    expected = "/path/to/file.hdr"
    assert envi_header(input_path) == expected


def test_envi_header_dat_extension():
    """Test envi_header with .dat extension."""
    input_path = "/path/to/file.dat"
    expected = "/path/to/file.hdr"
    assert envi_header(input_path) == expected


def test_envi_header_raw_extension():
    """Test envi_header with .raw extension."""
    input_path = "/path/to/file.raw"
    expected = "/path/to/file.hdr"
    assert envi_header(input_path) == expected


def test_envi_header_hdr_extension():
    """Test envi_header with .hdr extension."""
    input_path = "/path/to/file.hdr"
    expected = "/path/to/file.hdr"
    assert envi_header(input_path) == expected


def test_envi_header_no_extension():
    """Test envi_header with no extension."""
    input_path = "/path/to/file"
    expected = "/path/to/file.hdr"
    assert envi_header(input_path) == expected
