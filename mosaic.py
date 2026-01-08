
import argparse
import os
import uuid

from osgeo import gdal
import numpy as np


def process_file(f, ulx, lrx):
    ds = gdal.Open(f)
    band1 = ds.GetRasterBand(1).ReadAsArray()
    gt = ds.GetGeoTransform()
    width = ds.RasterXSize
    height = ds.RasterYSize
    uly = gt[3]
    lry = uly + gt[5] * height
    lon = np.linspace(ulx + gt[1] / 2, lrx - gt[1] / 2, width)
    lat = np.linspace(uly + gt[5] / 2, lry - gt[5] / 2, height)
    return ds, band1, gt, lon, lat

def create_clip(ds, gt, lon, lat, bounds, file):
    top, bottom, left, right = bounds
    driver = gdal.GetDriverByName('GTiff')
    rows = int(bottom + 1 - top)
    cols = int(right - left + 1)
    clip_ds = driver.Create(file, cols, rows, ds.RasterCount, gdal.GDT_Float32)
    new_gt = list(gt)
    new_gt[0] = -180 - (180 - lon[left]) if lon[left] > 180 else lon[left]
    new_gt[3] = lat[top]
    clip_ds.SetGeoTransform(new_gt)
    clip_ds.SetProjection(ds.GetProjection())
    for i in range(ds.RasterCount):
        band = ds.GetRasterBand(i + 1).ReadAsArray()
        clip_ds.GetRasterBand(i + 1).WriteArray(band[top:bottom + 1, left:right + 1])
        clip_ds.GetRasterBand(i + 1).SetNoDataValue(-9999)
    clip_ds.FlushCache()

def create_mosaic(input_files, output_file):
    west_files = []
    east_files = []

    for f in input_files:
        info = gdal.Info(f, format='json')
        ulx = info['cornerCoordinates']['upperLeft'][0]
        lrx = info['cornerCoordinates']['lowerRight'][0]
        print(f'File: {f}')
        print(f'  Upper Left Longitude: {ulx}')
        print(f'  Lower Right Longitude: {lrx}')


        left_out = f'/vsimem/{uuid.uuid4().hex}_left.tif'
        right_out = f'/vsimem/{uuid.uuid4().hex}_right.tif'

        if ulx < 180 and lrx > 180:
            print('  Crosses antimeridian, splitting\n')
            ds, band1, gt, lon, lat = process_file(f, ulx, lrx)
            lright = np.argwhere(lon >= 179.999)[0][0]
            lband = band1[:, :lright + 1]
            ltop, lbottom = np.argwhere(np.sum(lband == -9999, axis=1) != lband.shape[1])[[0, -1]].flatten()
            left_bounds = [ltop, lbottom, 0, lright]
            rleft = np.argwhere(lon >= 180.001)[0][0]
            rright = band1.shape[1] - 1
            rband = band1[:, rleft:]
            rtop, rbottom = np.argwhere(np.sum(rband == -9999, axis=1) != rband.shape[1])[[0, -1]].flatten()
            right_bounds = [rtop, rbottom, rleft, rright]
            for bounds, file in zip([left_bounds, right_bounds], [left_out, right_out]):
                create_clip(ds, gt, lon, lat, bounds, file)
            west_files.append(right_out)
            east_files.append(left_out)
        elif ulx > 180:
            print('  Lies east of east antimeridian, moving west\n')

            ds, band1, gt, lon, lat = process_file(f, ulx, lrx)
            rleft = np.argwhere(lon >= 180.001)[0][0]
            rright = band1.shape[1] - 1
            rband = band1[:, rleft:]
            rtop, rbottom = np.argwhere(np.sum(rband == -9999, axis=1) != rband.shape[1])[[0, -1]].flatten()
            right_bounds = [rtop, rbottom, rleft, rright]
            right_out = f'/vsimem/{uuid.uuid4().hex}_right.tif'
            create_clip(ds, gt, lon, lat, right_bounds, right_out)
            west_files.append(right_out)
        elif lrx <= 0:
            print('  Assigned to: WEST\n')
            west_files.append(f)
        elif ulx >= 0:
            print('  Assigned to: EAST\n')
            east_files.append(f)

    crosses_antimeridian = bool(west_files and east_files and west_files != east_files)
    print(f'Crosses antimeridian: {crosses_antimeridian}\n')

    translate_options = gdal.TranslateOptions(format="COG",
                                              creationOptions=["TILING_SCHEME=GoogleMapsCompatible"])

    def build_and_translate(files, out_path):
        vrt_path = f'/vsimem/{uuid.uuid4().hex}.vrt'
        vrt_options = gdal.BuildVRTOptions(resolution='average')
        vrt_ds = gdal.BuildVRT(vrt_path, files, options=vrt_options)
        gdal.Translate(out_path, vrt_ds, options=translate_options)
        print(f'Mosaic written to: {out_path}\n')

    if crosses_antimeridian and len(input_files) > 1:
        output_base, ext = os.path.splitext(output_file)
        west_output = f'{output_base}_2{ext}'

        print(f'Creating separate mosaics:\n  West: {west_output}\n  East: {output_file}\n')

        if west_files:
            print(f'Building west mosaic from {len(west_files)} files')
            build_and_translate(west_files, west_output)

        if east_files:
            print(f'Building east mosaic from {len(east_files)} files')
            build_and_translate(east_files, output_file)
    else:
        print('Creating single mosaic (no antimeridian crossing or only one input)\n')
        build_and_translate(input_files, output_file)


def main():
    parser = argparse.ArgumentParser(description="Export Cloud Optimized GeoTIFF mosaic")
    parser.add_argument('input', nargs='+', help='Input files')
    parser.add_argument('output', help='Output file')
    args = parser.parse_args()

    create_mosaic(args.input, args.output)

if __name__ == "__main__":
    main()