import argparse
from spectral.io import envi

import os
import scipy
import numpy as np
from utils import envi_header
from osgeo import gdal
import matplotlib.pyplot as plt
from scipy import signal
from scipy.ndimage import gaussian_filter
from apply_glt import single_image_ortho
import logging
import json
import glob
import shapely
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
import shapely.ops
from datetime import datetime, timedelta



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




def write_output_file(source_ds, output_img, output_file):
    driver = gdal.GetDriverByName('GTiff')
    driver.Register()
    outDataset = driver.Create(output_file,source_ds.RasterXSize,source_ds.RasterYSize,3,gdal.GDT_Byte,options = ['COMPRESS=LZW'])
    outDataset.SetProjection(source_ds.GetProjection())
    outDataset.SetGeoTransform(source_ds.GetGeoTransform())
    for n in range(1,4):
        outDataset.GetRasterBand(n).WriteArray(output_img[...,n-1])
        outDataset.GetRasterBand(n).SetNoDataValue(0)
    del outDataset



def plume_mask(input: np.array, plume_mask):

    y_locs = np.where(np.sum(plume_mask > 0, axis=1))[0]
    x_locs = np.where(np.sum(plume_mask > 0, axis=0))[0]

    plume_dat = input[y_locs[0]:y_locs[-1],x_locs[0]:x_locs[-1]]
    plume_dat[plume_mask[y_locs[0]:y_locs[-1],x_locs[0]:x_locs[-1]] == 0] = 0
    
    plume_dat = gauss_blur(plume_dat, 3, preserve_nans=False)
    local_output_mask = plume_dat > 50
    
    output_plume_mask = np.zeros(plume_mask.shape,dtype=bool)
    output_plume_mask[y_locs[0]:y_locs[-1],x_locs[0]:x_locs[-1]] = local_output_mask
    #output_plume_mask[signal.convolve2d(output_plume_mask,np.ones((10,10)),mode='same') > 10*10*0.25] = True
    
    labels, label_counts = scipy.ndimage.label(local_output_mask)
    out_labels = np.zeros(plume_mask.shape,dtype=int)
    out_labels[y_locs[0]:y_locs[-1],x_locs[0]:x_locs[-1]] = labels

    return output_plume_mask, out_labels


def gauss_blur(img, sigma, preserve_nans=False):
    
    V=img.copy()
    V[np.isnan(img)]=0
    VV=gaussian_filter(V,sigma=sigma)

    W=0*img.copy()+1
    W[np.isnan(img)]=0
    WW=gaussian_filter(W,sigma=sigma)

    img_smooth=VV/WW
    if preserve_nans:
        img_smooth[np.isnan(img)] = np.nan
    
    return img_smooth


def main(input_args=None):
    parser = argparse.ArgumentParser(description="Delineate/colorize plume")
    parser.add_argument('input_file', type=str,  metavar='INPUT', help='path to input image')   
    parser.add_argument('maskfile', type=str,  metavar='plume_mask', help='confining location of plume')
    parser.add_argument('gltfile', type=str,  metavar='glt file', help='confining location of plume')
    #parser.add_argument('igmfile', type=str,  metavar='igm file', help='lat,lon,elev of plume')
    parser.add_argument('output_file', type=str,  metavar='OUTPUT', help='path to output image')   
    parser.add_argument('-vmax', type=float, nargs=1, default=1500)
    parser.add_argument('-blur_sigma', type=float, default=0)
    args = parser.parse_args(input_args)
    

    # Load Data
    ds = gdal.Open(args.input_file,gdal.GA_ReadOnly)
    dat = ds.ReadAsArray().astype(np.float32)
    input_plume_mask = np.load(args.maskfile)

    # Load GLT
    glt_ds = gdal.Open(args.gltfile)
    glt = glt_ds.ReadAsArray().transpose((1,2,0))
    trans = glt_ds.GetGeoTransform()

    # Load IGM
    #igm_ds = gdal.Open(args.igmfile)
    #igm = igm_ds.ReadAsArray().transpose((1,2,0))

    # Create Mask
    dat[dat == ds.GetRasterBand(1).GetNoDataValue()] = np.nan
    plume, plume_labels = plume_mask(dat, input_plume_mask)

    # Make the output colored plume image
    rawdat = dat.copy()
    dat[np.logical_not(plume)] = np.nan
    dat = gauss_blur(dat, args.blur_sigma)   
    dat[np.logical_not(plume)] = 0

    dat /= 1500
    dat[dat > 1] = 1
    dat[dat < 0] = 0
    dat[dat == 0] = 0.01
    
    if args.blur_sigma > 0:
        dat[np.logical_not(plume)] = np.nan
        dat = gauss_blur(dat, args.blur_sigma)   
    
    colorized = np.zeros((dat.shape[0],dat.shape[1],3))
    colorized[plume] = plt.cm.plasma(dat[plume])[...,:3]
    colorized = np.round(colorized * 255).astype(np.uint8)

    colorized = single_image_ortho(colorized, glt)

    
    rawshape = (rawdat.shape[0],rawdat.shape[1],1)
    rawdat = single_image_ortho(rawdat.reshape(rawshape), glt)[...,0]
    plume_labels = single_image_ortho(plume_labels.reshape(rawshape), glt)[...,0]
    plume_labels[plume_labels == -9999] = 0
    rawdat[rawdat == -9999] = np.nan
    
    write_output_file(glt_ds, colorized, args.output_file)


    fid = os.path.basename(args.input_file).split('_')[0]
    date=fid[4:]
    time=fid.split('t')[-1]
    start_datetime = datetime.strptime(fid[4:], "%Y%m%dt%H%M%S")
    end_datetime = start_datetime + timedelta(seconds=1)

    start_datetime = start_datetime.strftime("%Y-%m-%dT%H:%M:%SZ")
    end_datetime = end_datetime.strftime("%Y-%m-%dT%H:%M:%SZ")

    #datetime = f'{date[:4]}-{date[4:6]}-{date[6:8]}T{time[:2]}:{time[2:4]}:{time[4:6]}Z'
    #datetime_e = f'{date[:4]}-{date[4:6]}-{date[6:8]}T{time[:2]}:{time[2:4]}:{str(int(time[4:6])+1).zfill(2)}Z'



    l1b_rad = glob.glob(f'/beegfs/store/emit/ops/data/acquisitions/{fid[4:12]}/{fid.split("_")[0]}/l1b/EMIT_L1B_RAD*_cnm.out')
    if len(l1b_rad) > 0:
        l1b_rad = os.path.basename(l1b_rad[0]).split('.')[0][:-20]
        l1b_link = f'https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL1BRAD.001/{l1b_rad}/{l1b_rad}.nc'
    else:
        l1b_link = f'Scene Not Yet Available'



    # Make output vector
    outdict = {"crs": {"properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"}, "type": "name"},"features":[],"name":"methane_metadata","type":"FeatureCollection" }
    un_labels = np.unique(plume_labels)[1:]
    background = np.round(np.nanstd(rawdat[plume_labels == 0]))
    loclist = []
    for lab in un_labels:
        
        maxval = np.nanmax(rawdat[plume_labels == lab])
        if np.sum(plume_labels == lab) < 5 or maxval < 200:
            continue

        #rawloc = np.where(np.logical_and(rawdat == maxval, plume_labels == lab))
        maxval = np.round(maxval)

        y_locs = np.where(np.sum(plume_labels == lab, axis=1))[0]
        x_locs = np.where(np.sum(plume_labels == lab, axis=0))[0]
        radius = np.sqrt( ((y_locs[0] - y_locs[-1])*trans[5])**2 + ((x_locs[0] - x_locs[-1])*trans[1])**2)/2.
        center_y = trans[3] + trans[5] * (y_locs[0] + (y_locs[-1]-y_locs[0])/2)
        center_x = trans[0] + trans[1] * (x_locs[0] + (x_locs[-1]-x_locs[0])/2)

        point = shapely.geometry.Point(center_x, center_y)
        circle = shapely.geometry.polygon.Polygon(point.buffer(radius))


        #loclist.append(rawloc)
        #loc_y = trans[3] + trans[5]*rawloc[0][0]
        #loc_x = trans[0] + trans[1]*rawloc[1][0]
        #loc_z = 0

        #lloc = igm[rawloc[0][0], rawloc[1][0],:].tolist()

        #loc_res = {"geometry": {"coordinates": [loc_x, loc_y, loc_z], "type": "Point"},
        #           "type": "Feature",
        #           "properties": {"UTC Time Observed": start_datetime, 
        #                          "map_endtime": end_datetime,
        #                          "Max Plume Concentration (ppm m)": maxval,
        #                          "Concentration Uncertainty (ppm m)": background,
        #                          "Scene FID": fid,
        #                          "L1B Radiance Download": l1b_link,
        #                          "label": f'UTC Time Observed: {start_datetime}\nMax Plume Concentration (ppm m): {maxval}\nConcentration Uncertainty (ppm m): {background}'}}
        #                          


        #loc_res = {"geometry": {"coordinates": [list(circle.exterior.coords)], "type": "Polygon"},
        loc_res = {"geometry": {"coordinates": [center_x, center_y, 0.0], "type": "Point"},
                   "vis_style": {"radius": radius},
                   "type": "Feature",
                   "properties": {"UTC Time Observed": start_datetime, 
                                  "map_endtime": end_datetime,
                                  "Max Plume Concentration (ppm m)": maxval,
                                  "Concentration Uncertainty (ppm m)": background,
                                  "Scene FID": fid,
                                  "L1B Radiance Download": l1b_link
                                  }}
        
        outdict['features'].append(loc_res)

    with open(os.path.splitext(args.output_file)[0] + '.json', 'w') as fout:
        fout.write(json.dumps(outdict, cls=SerialEncoder, indent=2, sort_keys=True)) 
    

    #plt.imshow(colorized.astype(np.float32)/255.)
    #for feat in outdict['features']:
    #    plt.scatter((feat['geometry']['coordinates'][0] - trans[0])/trans[1], (feat['geometry']['coordinates'][1] - trans[3]) / trans[5], edgecolors='red',s=10, facecolors='none')
    ##for loc in loclist:
    ##    plt.scatter(loclist[0],loclist[1], edgecolors='red',s=80, facecolors='none')
    #plt.ylim([750,1250])
    #plt.xlim([1500,2000])
    #plt.savefig(f'test_{fid}.png',dpi=300)



if __name__ == '__main__':
    main()

