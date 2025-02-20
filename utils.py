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

import os
import datetime
import json

import numpy as np
from osgeo import gdal

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

def convert_to_cog(input_file, output_file,product_metadata,software_build_version,product_version):

    metadata = {}
    metadata['keywords'] = "Imaging Spectroscopy, minerals, EMIT, dust, radiative forcing"
    metadata['sensor'] = "EMIT (Earth Surface Mineral Dust Source Investigation)"
    metadata['instrument'] = "EMIT"
    metadata['platform'] = "ISS"
    metadata['Conventions'] = "CF-1.63"
    metadata['institution'] = "NASA Jet Propulsion Laboratory/California Institute of Technology"
    metadata['license'] = "https://science.nasa.gov/earth-science/earth-science-data/data-information-policy/"
    metadata['naming_authority'] = "LPDAAC"
    metadata['date_created'] = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ")
    metadata['keywords_vocabulary'] = "NASA Global Change Master Directory (GCMD) Science Keywords"
    metadata['stdname_vocabulary'] = "NetCDF Climate and Forecast (CF) Metadata Convention"
    metadata['creator_name'] = "Jet Propulsion Laboratory/California Institute of Technology"
    metadata['creator_url'] = "https://earth.jpl.nasa.gov/emit/"
    metadata['project'] = "Earth Surface Mineral Dust Source Investigation"
    metadata['project_url'] = "https://earth.jpl.nasa.gov/emit/"
    metadata['publisher_name'] = "NASA LPDAAC"
    metadata['publisher_url'] = "https://lpdaac.usgs.gov"
    metadata['publisher_email'] = "lpdaac@usgs.gov"
    metadata['identifier_product_doi_authority'] = "https://doi.org"
    metadata['software_build_version'] = software_build_version
    metadata['product_version'] = product_version

    metadata['description'] = product_metadata['description']

    ds_mem = gdal.Translate('', input_file, format='MEM')
    ds_mem.SetMetadata(metadata)

    band = ds_mem.GetRasterBand(1)
    band.SetDescription(product_metadata['name'])
    band.SetMetadataItem("UNITS",product_metadata['units'])
    band.SetMetadataItem("fill_value","-1")

    band.FlushCache()

    translate_options = gdal.TranslateOptions(format='COG')
    gdal.Translate(output_file, ds_mem, options=translate_options)

    ds_mem = None
