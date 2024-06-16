#! /usr/bin/env python
#
#  Copyright 2024 California Institute of Technology
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
import subprocess
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from osgeo import gdal
import os
import json
from scrape_refine_upload import spatial_temporal_filter
import pandas as pd
from shapely.geometry import Polygon
import glob
import apply_glt



def multifile_rawspace_conversion(glt_files, coordinates):
    rawspace_coords = []
    y_offset = 0
    for _g, glt_file in enumerate(glt_files):
        ds = gdal.Open(glt_file)
        glt = ds.ReadAsArray()
        trans = ds.GetGeoTransform()

        for ind in coordinates:
            glt_ypx = int(round((ind[1] - trans[3])/ trans[5]))
            glt_xpx = int(round((ind[0] - trans[0])/ trans[1]))

            if glt_ypx < 0 or glt_ypx >= glt.shape[0] or glt_xpx < 0 or glt_xpx >= glt.shape[1]:
                continue

            lglt = glt[glt_ypx, glt_xpx,:].tolist()
            lglt[1] += y_offset
            lglt.append(_g)
            rawspace_coords.append(lglt)
        y_offset += np.max(glt[...,1])

    return rawspace_coords


def get_mf_ort_file(fid, sourcedir):
    return os.path.join(sourcedir, f"{fid}_mf_ort")


def get_mf_file(fid, sourcedir):
    return os.path.join(sourcedir, f"{fid}_mf")


def get_glt_file(fid):
    glt_file = sorted(glob.glob(f'/beegfs/store/emit/ops/data/acquisitions/{fid[4:12]}/{fid.split("_")[0]}/l1b/*_l1b_glt_*.img'))[-1]
    return glt_file


def get_min_file(fid):
    min_file = sorted(glob.glob(f'/beegfs/store/emit/ops/data/acquisitions/{fid[4:12]}/{fid.split("_")[0]}/l2b/*_l2b_abun_*.img'))[-1]
    return min_file


def get_cover_file(fid):
    cover_file = sorted(glob.glob(f'/beegfs/store/emit/ops/data/acquisitions/{fid[4:12]}/{fid.split("_")[0]}/l3/*_l3_cover_*.img'))[-1]
    return cover_file


def lonlat_from_coords(coords):
    lons = np.array([x[0] for x in coords])
    lats = np.array([x[1] for x in coords])
    return lons, lats


def read_from_bounds(ds, coords, buffer=50):
    lons, lats = lonlat_from_coords(coords)
    trans = ds.GetGeoTransform()
    x_off = int(np.round(np.min(lons) - trans[0])/trans[1])
    x_size = int(np.round((np.max(lons) - np.min(lons))/trans[1]))

    y_off = int(np.round(np.max(lats) - trans[3])/trans[5])
    y_size = int(np.round((np.min(lats) - np.max(lats))/trans[5]))

    x_off = max(x_off - buffer,0)
    x_size = min(x_size + buffer, ds.RasterXSize - x_off)

    y_off = max(y_off - buffer,0)
    y_size = min(y_size + buffer, ds.RasterYSize - y_off)

    dat = ds.ReadAsArray(x_off, y_off, x_size, y_size)
    if len(dat.shape) == 3:
        dat = np.moveaxis(dat, 0, 2)
    else:
        dat = np.expand_dims(dat, 2)
    return dat, x_off, y_off


def read_raw_from_bounds(files, coords, buffer=50):
    dat = []
    for _f, file in enumerate(files):
        ds = gdal.Open(file)
        dat.append(ds.ReadAsArray())
    dat = np.concatenate(dat, axis=0)
    x_off = int(np.round(np.min(coords[...,0])) - buffer)
    
    return 


def add_mf(fids, coord_list, sourcedir):
    if len(fids) == 0:
        return 
    mf_files = [get_mf_ort_file(fid, sourcedir) for fid in fids]
    vrt_ds = gdal.BuildVRT('', mf_files)
    dat, x_off, y_off = read_from_bounds(vrt_ds, coord_list)

    #mf_files = sorted([get_mf_file(fid, sourcedir) for fid in fids])
    #rawspace_coords = multifile_rawspace_conversion([get_glt_file(x) for x in fids], coord_list)
    #dat = read_raw_from_bounds(mf_files, rawspace_coords)


    dat /= 1500
    dat[dat > 1] = 1
    dat[dat < 0] = 0
    dat[dat == 0] = 0.01
    plt.imshow(dat, cmap='plasma')
    plt.axis('off')
    plot_plume(coord_list, x_off, y_off, vrt_ds.GetGeoTransform())


def add_minerology(fids, coord_list, scratchdir, group='1'):

    min_ort_files = []
    for fid in fids:
        glt_file = get_glt_file(fid)
        min_file = get_min_file(fid)
        min_ort_file = os.path.join(scratchdir, f"{fid}_min_ort")

        apply_glt.main([glt_file, min_file, min_ort_file])
        min_ort_files.append(min_ort_file)
    
    vrt_ds = gdal.BuildVRT('', min_ort_files)
    dat, x_off, y_off = read_from_bounds(vrt_ds, coord_list)
    if group == '1':
        min = dat[...,1]
    elif group == '2':
        min = dat[...,3]

    mask = min <= 0
    col_min1 = plt.get_cmap('tab20')(min/np.max(min))
    col_min1[mask] = [0,0,0,0]
    
    plt.imshow(min)
    plt.axis('off')
    plot_plume(coord_list, x_off, y_off, vrt_ds.GetGeoTransform())


def add_cover(fids, coord_list, scratchdir, group='1'):

    cover_ort_files = []
    for fid in fids:
        glt_file = get_glt_file(fid)
        cover_file = get_cover_file(fid)
        cover_ort_file = os.path.join(scratchdir, f"{fid}_cover_ort")

        apply_glt.main([glt_file, cover_file, cover_ort_file])
        cover_ort_files.append(cover_ort_file)
    
    vrt_ds = gdal.BuildVRT('', cover_ort_files)
    dat, x_off, y_off = read_from_bounds(vrt_ds, coord_list)

    dat[dat < 0] = np.nan
    dat = dat - np.nanpercentile(dat, 2, axis=(0,1))[np.newaxis,np.newaxis,:]
    dat = dat / np.nanpercentile(dat, 98, axis=(0,1))[np.newaxis,np.newaxis,:]
    
    plt.imshow(dat)
    plt.axis('off')
    plot_plume(coord_list, x_off, y_off, vrt_ds.GetGeoTransform())
 

def plot_plume(coords, x_off, y_off, trans):
    lons, lats = lonlat_from_coords(coords)
    x_pos = np.array([int(np.round((x - trans[0])/trans[1] - x_off)) for x in lons])
    y_pos = np.array([int(np.round((x - trans[3])/trans[5] - y_off)) for x in lats])
    plt.plot(x_pos, y_pos, c='w',ls='--')



    

def build_pdf_page(plume_info, coverage, coverage_df, sourcedir, scratchdir, outfile):
    with PdfPages(outfile) as pdf:
        fig = plt.figure(figsize=(17, 22))
        gs = fig.add_gridspec(6, 6, hspace=0.5, wspace=0.5)


        loc_coords = plume_info['geometry']['coordinates'][0]


        # Add current, previous, and next matched filter outputs

        # Previous
        ax = fig.add_subplot(gs[0:2, 0:2])
        dt = pd.to_datetime(plume_info['properties']['Time Range Start']) - pd.Timedelta('1 minute')
        dts = dt.strftime('%Y-%m-%dT%H:%M:%SZ')
        subset_features = spatial_temporal_filter(coverage, coverage_df, loc_coords, '2020-01-01T00:00:00', dts)
        subset_features = [subset_features['features'][-1]] #todo - be more clever about this
        fids = [x['properties']['fid'] for x in subset_features['features']]
        add_mf(fids, loc_coords, sourcedir)
        plt.title('Previous MF; ' + fids[-1])


        # Current
        ax = fig.add_subplot(gs[2:4, 0:2])
        add_mf(plume_info['properties']['fids'], loc_coords, sourcedir)
        plt.title('Current MF; ' + plume_info['properties']['fids'][0])


        # Future
        ax = fig.add_subplot(gs[4:6, 0:2])
        dt = pd.to_datetime(plume_info['properties']['Time Range End']) + pd.Timedelta('1 minute')
        dts = dt.strftime('%Y-%m-%dT%H:%M:%SZ')
        subset_features = spatial_temporal_filter(coverage, coverage_df, loc_coords, dts, '2040-01-01T00:00:00')
        subset_features = [subset_features['features'][0]] #todo - be more clever about this
        fids = [x['properties']['fid'] for x in subset_features['features']]
        add_mf(fids, loc_coords, sourcedir)
        plt.title('Next MF; ' + fids[-1])


        # Add minerology
        ax = fig.add_subplot(gs[0:2, 2:4])
        add_minerology(fids, loc_coords, scratchdir, group='1')
        plt.title('Group 1 Minerals')

        ax = fig.add_subplot(gs[2:4, 2:4])
        add_minerology(fids, loc_coords, scratchdir, group='1')
        plt.title('Group 2 Minerals')

        # fractional cover
        ax = fig.add_subplot(gs[4:6, 2:4])
        add_cover(fids, loc_coords, scratchdir, group='1')
        plt.title('Fractional Cover')

        # Add radiance and reflectance plots




def main(input_args=None):
    parser = argparse.ArgumentParser(description="merge jsons")
    parser.add_argument('--out_dir', type=str,  metavar='OUTPUT_DIR', help='output directory')   
    parser.add_argument('--scratch_dir', type=str,  metavar='OUTPUT_DIR', help='output directory')   
    parser.add_argument('--source_dir', type=str,  default='/beegfs/scratch/brodrick/methane/methane_20230813', metavar='INPUT_DIR', help='input directory')   
    parser.add_argument('--type', type=str,  choices=['ch4','co2'], default='ch4')   
    parser.add_argument('--database_config', type=str,  default='/beegfs/store/emit//ops/repos/emit-main/emit_main/config/ops_sds_config.json')   
    parser.add_argument('--loglevel', type=str, default='DEBUG', help='logging verbosity')    
    parser.add_argument('--logfile', type=str, default=None, help='output file to write log to')    
    parser.add_argument('--continuous', action='store_true', help='run continuously')    
    parser.add_argument('--track_coverage_file', default='/beegfs/scratch/brodrick/emit/emit-visuals/track_coverage_pub.json')
    args = parser.parse_args(input_args)

    if os.path.isdir(args.out_dir) is False:
        subprocess.call(f'mkdir {args.out_dir}',shell=True)
    if os.path.isdir(args.scratch_dir) is False:
        subprocess.call(f'mkdir {args.scratch_dir}',shell=True)

    previous_annotation_file = os.path.join(args.out_dir, "previous_manual_annotation.json")

    annotation = json.load(open(previous_annotation_file, 'r'))
    coverage = json.load(open(args.track_coverage_file, 'r'))

    coverage_df = pd.json_normalize(coverage['features'])
    coverage_df['geometry.coordinates'] = coverage_df['geometry.coordinates'].apply(lambda s: Polygon(s[0]) )
    coverage_df['properties.start_time'] = pd.to_datetime(coverage_df['properties.start_time'])
    coverage_df['properties.end_time'] = pd.to_datetime(coverage_df['properties.end_time'])

    for plume in annotation['features']:
        outfile = os.path.join(args.out_dir, f"{plume['date']}_{plume['time']}.pdf")
        build_pdf_page(plume, coverage, coverage_df, args.sourcedir, args.scratch_dir, outfile)
        exit()




 
if __name__ == '__main__':
    main()