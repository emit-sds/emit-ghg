#! /usr/bin/env python
#
#  Copyright 2023 California Institute of Technology
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
# Authors: Philip G. Brodrick, philip.brodrick@jpl.nasa.gov

import netCDF4 as nc
import numpy as np
import os


class SpectralMetadata:
    def __init__(self, wavelengths, fwhm):
        """
        Initializes the SpectralMetadata object.

        Args:
            wavelengths (numpy.ndarray): Array of wavelength values.
            fwhm (numpy.ndarray): Array of full-width half-maximum values.
        """
        self.wavelengths = wavelengths
        self.wl = wavelengths
        self.fwhm = fwhm


def open_emit_rdn(input_file):
    """
    Opens an EMIT radiance NetCDF file and extracts the spectral metadata and radiance data.

    Args:
        input_file (str): Path to the NetCDF file.

    Returns:
        tuple: A tuple containing:
            - SpectralMetadata: An object containing the wavelengths and FWHM.
            - numpy.ndarray: The radiance data as a numpy array with shape (lines, bands, samples).
    """
    ds = nc.Dataset(input_file)
    wl = ds['sensor_band_parameters']['wavelengths'][:]
    fwhm = ds['sensor_band_parameters']['fwhm'][:]
    rdn = np.array(ds['radiance'][:])

    meta = SpectralMetadata(wl, fwhm)

    return meta, rdn


def open_airborne_rdn(input_file):
    """
    Opens an Airborne radiance NetCDF file and extracts the spectral metadata and radiance data.

    Args:
        input_file (str): Path to the NetCDF file.

    Returns:
        tuple: A tuple containing:
            - SpectralMetadata: An object containing the wavelengths and FWHM.
            - numpy.ndarray: The radiance data as a numpy array with shape (lines, bands, samples).
    """
    ds = nc.Dataset(input_file)
    wl = ds['radiance']['wavelength'][:]
    fwhm = ds['radiance']['fwhm'][:]

    rdn = np.transpose(ds['radiance']['radiance'][:], (1, 2, 0))

    meta = SpectralMetadata(wl, fwhm)

    return meta, rdn


def open_netcdf_radiance(input_file):
    """
    Opens a NetCDF radiance file and extracts the metadata and data.
    Automatically detects whether it's EMIT, AV3, or AV5 format.

    Args:
        input_file (str): Path to the NetCDF file.

    Raises:
        ValueError: If the file type cannot be determined.
        FileNotFoundError: If the file does not exist.

    Returns:
        tuple: A tuple containing:
            - SpectralMetadata: An object containing the wavelengths and FWHM.
            - numpy.ndarray: The radiance data as a numpy array.
    """
    if not os.path.exists(input_file):
        raise FileNotFoundError(f'{input_file} not found.')

    input_filename = os.path.basename(input_file)
    input_filename_lower = input_filename.lower()

    if 'emit' in input_filename_lower and ('rad' in input_filename_lower or 'rdn' in input_filename_lower):
        return open_emit_rdn(input_file)

    is_airborne_like = any(tag in input_filename_lower for tag in ['av3', 'av5', 'ang', 'prm', 'prism'])
    if is_airborne_like and 'rdn' in input_filename_lower:
        return open_airborne_rdn(input_file)

    raise ValueError(f'Unknown or unsupported NetCDF radiance file type for {input_file}')
