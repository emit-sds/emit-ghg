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

import argparse
import numpy as np
from osgeo import gdal
import matplotlib.pyplot as plt


def write_output_file(source_ds, output_img, output_file):
    driver = gdal.GetDriverByName('GTiff')
    driver.Register()
    outDataset = driver.Create(output_file,source_ds.RasterXSize,source_ds.RasterYSize,3,gdal.GDT_Byte,options = ['COMPRESS=LZW'])
    outDataset.SetProjection(source_ds.GetProjection())
    outDataset.SetGeoTransform(source_ds.GetGeoTransform())
    for n in range(1,4):
        if len(output_img.shape) == 2:
            outDataset.GetRasterBand(n).WriteArray(output_img)
        else:
            outDataset.GetRasterBand(n).WriteArray(output_img[...,n-1])
        outDataset.GetRasterBand(n).SetNoDataValue(0)
    del outDataset



def main(input_args=None):
    parser = argparse.ArgumentParser(description="Robust MF")
    parser.add_argument('input_file', type=str,  metavar='INPUT', help='path to input image')   
    parser.add_argument('output_file', type=str,  metavar='OUTPUT', help='path to input image')   
    parser.add_argument('bounds', type=float,  nargs=2, metavar='SCALING RANGE', help='path to input image')   
    parser.add_argument('--cmap', type=str,  default=None, choices=['plasma','YlOrRd', 'viridis', 'RdBu_r'], metavar='COLOR_SCALE', help='color scale to apply')   
    args = parser.parse_args(input_args)


    ds = gdal.Open(args.input_file,gdal.GA_ReadOnly)
    dat = ds.ReadAsArray().astype(np.float32)
    dat[dat == ds.GetRasterBand(1).GetNoDataValue()] = np.nan

    print(args.cmap)
    print(f'scsaling bounds: {args.bounds}')
    if args.cmap is None:
        dat -= args.bounds[0]
        dat /= (args.bounds[1] - args.bounds[0])

        dat *= 255
        dat[dat > 255] = 255
        dat[dat < 1] = 1
        print(np.sum(dat[np.isfinite(dat)] == 0))
        dat[np.isnan(dat)] = 0
    else:

        dat -= args.bounds[0]
        dat /= (args.bounds[1] - args.bounds[0])
        isnan = np.isnan(dat)
        dat[dat <= 0] = 0.01
        dat[isnan] = 0
        mask = dat == 0
        if args.cmap == 'plasma':
            dat = plt.cm.plasma(dat)[...,:3]
        elif args.cmap == 'YlOrRd':
            dat = plt.cm.YlOrRd(dat)[...,:3]
        elif args.cmap == 'viridis':
            dat = plt.cm.viridis(dat)[...,:3]
        elif args.cmap == 'RdBu_r':
            dat = plt.cm.RdBu_r(dat)[...,:3]
        dat = np.round(dat * 255).astype(np.uint8)
        dat[np.logical_not(mask),:] = np.maximum(1, dat[np.logical_not(mask),:])
        dat[isnan,:] = 0

    write_output_file(ds, dat, args.output_file)


if __name__ == '__main__':
    main()
