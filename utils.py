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
import numpy as np
import json
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

   
def convert_to_cog(input_file, output_file,metadata = None):
    src_ds = gdal.Open(input_file)
    if src_ds is None:
        raise ValueError(f"Unable to open input file: {input_file}")

    options = [
        'COMPRESS=LZW',
        'TILED=YES',
        'COPY_SRC_OVERVIEWS=YES'
    ]

    gdal.Translate(output_file, src_ds, creationOptions=options)
    os.system(f"gdaladdo -minsize 900 {output_file}")

    src_ds = None
    
    if metadata:
        in_ds = gdal.Open(output_file, gdal.GA_Update)
        if in_ds is None:
            raise ValueError(f"Unable to open output file for updating: {output_file}")

        in_meta = in_ds.GetMetadata()

        if 'Band_1' in in_meta:
            del in_meta['Band_1']
        in_meta["DESCRIPTION"] =  metadata['description']   
        in_ds.SetMetadata(in_meta)
        
        band = in_ds.GetRasterBand(1)
        band.SetDescription(metadata['name'])
        band.SetMetadataItem("UNITS",metadata['units'])
        band.FlushCache()
        del band
        del in_ds
        
        