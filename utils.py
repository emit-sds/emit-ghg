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

from spectral.io import envi

import os
import numpy as np
import json
import pdb

def envi_header(inputpath):
    """
    Convert a envi binary/header path to a header, handling extensions
    Args:
        inputpath: path to envi binary file
    Returns:
        str: the header file associated with the input reference.

    """
    if os.path.splitext(inputpath)[-1] == '.img' or os.path.splitext(inputpath)[-1] == '.dat' or os.path.splitext(inputpath)[-1] == '.raw':
        # headers could be at either filename.img.hdr or filename.hdr.  Check both, return the one that exists if it
        # does, if not return the latter (new file creation presumed).
        hdrfile = os.path.splitext(inputpath)[0] + '.hdr'
        if os.path.isfile(hdrfile):
            return hdrfile
        elif os.path.isfile(inputpath + '.hdr'):
            return inputpath + '.hdr'
        return hdrfile
    elif os.path.splitext(inputpath)[-1] == '.hdr':
        return inputpath
    else:
        return inputpath + '.hdr'


def write_bil_chunk(dat, outfile, line, shape, dtype = 'float32'):
    """
    Write a chunk of data to a binary, BIL formatted data cube.
    Args:
        dat: data to write
        outfile: output file to write to
        line: line of the output file to write to
        shape: shape of the output file
        dtype: output data type

    Returns:
        None
    """
    outfile = open(outfile, 'rb+')
    outfile.seek(line * shape[1] * shape[2] * np.dtype(dtype).itemsize)
    outfile.write(dat.astype(dtype).tobytes())
    outfile.close()


class SerialEncoder(json.JSONEncoder):
    """Encoder for json to help ensure json objects can be passed to the workflow manager.
    """

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        else:
            return super(SerialEncoder, self).default(obj)

class ReadAbstractDataSet():
    """Wrapper class to enable reading from multiple input data types. Currently supports ENVI and netCDF4.
    netCDF4 is assumed if the file ends in .nc, otherwise it is assumed to be ENVI
    The netcdf_ inputs are used when the input file is netCDF4 and the envi_ inputs are used when it is ENVI.

    In order to be consistent with ENVI, the radiance data from netCDF4 files is transposed so that it matches
    the radiance data read from ENVI.
    """
    def __init__(self, filename, netcdf_group = None, netcdf_key = None, envi_interleave = None):
        self.filename = filename
        
        # Determine the file type: ENVI or netCDF4
        if filename[-3:] == '.nc':
            self.filetype = 'netCDF4'
            if netcdf_key is None:
                raise ValueError(f'Keyword netcdf_key must be provided for netCDF4 files.')
        elif os.path.exists(envi_header(filename)):
            self.filetype = 'ENVI'
            if envi_interleave is None:
                raise ValueError(f'Keyword envi_interleave must be provided for ENVI files.')
        else:
            raise ValueError(f'File type of {filename} not recognized.')
        
        if self.filetype == 'ENVI':
            self.ds = envi.open(envi_header(filename), image = filename)
            self.memmap = self.ds.open_memmap(interleave=envi_interleave,writeable=False)
            self.metadata = self.ds.metadata

            wavelengths = None
            fwhm = None
            if 'wavelength' in self.ds.metadata:
                wavelengths = np.array([float(x) for x in self.ds.metadata['wavelength']])
                fwhm = np.array([float(x) for x in self.ds.metadata['fwhm']])
            self.metadata['wavelength'] = wavelengths
            self.metadata['fwhm'] = fwhm
        
        if self.filetype == 'netCDF4':
            import xarray as xr 
            self.ds = xr.open_dataset(self.filename, engine = 'netcdf4', group = netcdf_group)

            self.data = self.ds[netcdf_key].values
            
            # Transpose the radiance to match the way ENVI reads it in
            if netcdf_key == 'radiance':
                self.data = np.ascontiguousarray(np.transpose(self.data, [0,2,1]))
            
            if len(self.data.shape) == 3:
                l, b, s = self.data.shape
            else:
                l, b = self.data.shape
                s = 0

            wvl = xr.open_dataset(self.filename, engine = 'netcdf4', group = 'sensor_band_parameters')
            try:
                wavelengths = wvl['wavelengths'].values
                fwhm = wvl['fwhm'].values
            except:
                wavelengths = None
                fwhm = None

            self.metadata = {'wavelength': wavelengths,
                             'fwhm': fwhm,
                             'bands': b,
                             'lines': l,
                             'samples': s,
                             'byte order': 0,
                             'header offset': 0,
                             'file_type': 'ENVI Standard',
                             'sensor type': 'unknown',
                             'band names': [f'channel_{x:d}' for x in range(b)]}

    def __getitem__(self, key):
        if self.filetype == 'ENVI':
            return self.memmap[key]
        if self.filetype == 'netCDF4':
            return self.data[key]

class WriteAbstractDataSet():
    """Wrapper class to write to arbitrary output file formats. Currenlty only supports ENVI.
    """
    def __init__(self, filename, outmeta = None):
        self.filename = filename
        
        if filename[-3:] == '.nc':
            self.filetype = 'netCDF4'
            raise NotImplementedError(f'Writing netCDF4 files is not supported')
        else:
            self.filetype = 'ENVI'
        
        if self.filetype == 'ENVI':
            ds = envi.create_image(envi_header(filename), outmeta, force = True, ext = '')
            del ds
    
    def write(self, *input_tuple):
        data, line, shape = input_tuple
        if self.filetype == 'ENVI':
            write_bil_chunk(data, self.filename, line, shape)
