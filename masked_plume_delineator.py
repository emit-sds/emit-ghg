import argparse
from spectral.io import envi

import os
import scipy
import numpy as np
from utils import envi_header
from osgeo import gdal, osr, ogr
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
import subprocess



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




def write_science_cog(output_img, output_file, geotransform, projection):
    tmp_file = os.path.splitext(output_file)[0] + '_tmp.tif'
    driver = gdal.GetDriverByName('GTiff')
    driver.Register()
    outDataset = driver.Create(tmp_file,output_img.shape[1],output_img.shape[0],1,gdal.GDT_Float32,options = ['COMPRESS=LZW'])
    outDataset.GetRasterBand(1).WriteArray(output_img)
    outDataset.GetRasterBand(1).SetNoDataValue(-9999)
    outDataset.SetProjection(projection)
    outDataset.SetGeoTransform(geotransform)
    del outDataset

    subprocess.call(f'sh /home/brodrick/bin/cog.sh {tmp_file} {output_file}',shell=True)
    subprocess.call(f'rm {tmp_file}',shell=True)
    



def write_output_file(source_ds, output_img, output_file):
    driver = gdal.GetDriverByName('GTiff')
    driver.Register()
    if len(output_img.shape) == 2:
        outDataset = driver.Create(output_file,source_ds.RasterXSize,source_ds.RasterYSize,1,gdal.GDT_Byte,options = ['COMPRESS=LZW'])
        outDataset.GetRasterBand(1).WriteArray(output_img)
    else:
        outDataset = driver.Create(output_file,source_ds.RasterXSize,source_ds.RasterYSize,3,gdal.GDT_Byte,options = ['COMPRESS=LZW'])
        for n in range(1,4):
            outDataset.GetRasterBand(n).WriteArray(output_img[...,n-1])
            outDataset.GetRasterBand(n).SetNoDataValue(0)

    outDataset.SetProjection(source_ds.GetProjection())
    outDataset.SetGeoTransform(source_ds.GetGeoTransform())
    del outDataset



def plume_mask(input: np.array, pm, style='ch4'):

    y_locs = np.where(np.sum(pm > 0, axis=1))[0]
    x_locs = np.where(np.sum(pm > 0, axis=0))[0]

    plume_dat = input[y_locs[0]:y_locs[-1],x_locs[0]:x_locs[-1]].copy()
    plume_dat[pm[y_locs[0]:y_locs[-1],x_locs[0]:x_locs[-1]] == 0] = 0
    
    if style == 'ch4':
        plume_dat = gauss_blur(plume_dat, 3, preserve_nans=False)
        local_output_mask = plume_dat > 50
    else:
        plume_dat = gauss_blur(plume_dat, 5, preserve_nans=False)
        local_output_mask = plume_dat > 10000
    
    output_plume_mask = np.zeros(pm.shape,dtype=bool)
    output_plume_mask[y_locs[0]:y_locs[-1],x_locs[0]:x_locs[-1]] = local_output_mask
    #output_plume_mask[signal.convolve2d(output_plume_mask,np.ones((10,10)),mode='same') > 10*10*0.25] = True
    
    labels, label_counts = scipy.ndimage.label(local_output_mask)
    out_labels = np.zeros(pm.shape,dtype=int)
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
    parser.add_argument('gltfile', type=str,  metavar='glt file', help='confining location of plume')
    #parser.add_argument('igmfile', type=str,  metavar='igm file', help='lat,lon,elev of plume')
    parser.add_argument('output_file', type=str,  metavar='OUTPUT', help='path to output image')   
    parser.add_argument('maskfiles', type=str,  nargs='+', metavar='plume_mask', help='confining location of plume')
    parser.add_argument('-output_mask', type=str,  default=None, metavar='output mask', help='path to output mask image')   
    parser.add_argument('-vmax', type=float, nargs=1, default=1500)
    parser.add_argument('-blur_sigma', type=float, default=0)
    parser.add_argument('-daac_demo_dir', type=str, default=None)
    args = parser.parse_args(input_args)
    

    # Load Data
    ds = gdal.Open(args.input_file,gdal.GA_ReadOnly)
    dat = ds.ReadAsArray().astype(np.float32)
    input_plume_mask = None
    for mask_file in args.maskfiles:
        if input_plume_mask is None:
            input_plume_mask = np.load(mask_file)
        else:
            input_plume_mask = input_plume_mask + np.load(mask_file)
    input_plume_mask = input_plume_mask >= 1

    # Load GLT
    glt_ds = gdal.Open(args.gltfile)
    glt = glt_ds.ReadAsArray().transpose((1,2,0))
    trans = glt_ds.GetGeoTransform()

    # Load IGM
    #igm_ds = gdal.Open(args.igmfile)
    #igm = igm_ds.ReadAsArray().transpose((1,2,0))

    # Create Mask
    if args.daac_demo_dir is not None:
        out_l2b_name = os.path.join(args.daac_demo_dir, 'l2b', os.path.splitext(os.path.basename(args.gltfile).replace('glt','ch4mf'))[0] + '_mv0.tif')
        write_science_cog(single_image_ortho(dat.reshape((dat.shape[0],dat.shape[1],1)), glt)[...,0], out_l2b_name, glt_ds.GetGeoTransform(), glt_ds.GetProjection())

    dat[dat == ds.GetRasterBand(1).GetNoDataValue()] = np.nan
    plume, plume_labels = plume_mask(dat, input_plume_mask)

    # Make the output colored plume image
    rawdat = dat.copy()
    background = np.round(np.nanstd(rawdat[np.logical_and(np.logical_not(plume), rawdat != -9999) ]))
    rawdat[np.logical_not(plume)] = -9999
    print(f'rawdat 0: {np.sum(rawdat == 0)}')
    dat[np.logical_not(plume)] = np.nan
    dat = gauss_blur(dat, args.blur_sigma)   
    dat[np.logical_not(plume)] = 0
    print(f'rawdat 0: {np.sum(rawdat == 0)}')

    dat /= 1500
    dat[dat > 1] = 1
    dat[dat < 0] = 0
    dat[dat == 0] = 0.01
    
    if args.blur_sigma > 0:
        dat[np.logical_not(plume)] = np.nan
        dat = gauss_blur(dat, args.blur_sigma)   
    
    colorized = np.zeros((dat.shape[0],dat.shape[1],3))
    colorized[plume,:] = plt.cm.plasma(dat[plume])[...,:3]
    colorized = np.round(colorized * 255).astype(np.uint8)
    colorized[plume,:] = np.maximum(1, colorized[plume,:])

    colorized = single_image_ortho(colorized, glt)

    
    rawshape = (rawdat.shape[0],rawdat.shape[1],1)
    print(f'rawdat 0: {np.sum(rawdat == 0)}')
    rawdat = single_image_ortho(rawdat.reshape(rawshape), glt)[...,0]
    print(f'rawdat 0: {np.sum(rawdat == 0)}, {rawdat[1570, 934]}')
    plume_labels = single_image_ortho(plume_labels.reshape(rawshape), glt)[...,0]


    plume_labels[plume_labels == -9999] = 0
    rawdat[rawdat == -9999] = np.nan
    print(f'rawdat 0: {np.sum(rawdat == 0)}, {rawdat[1570, 934]}')
    #rawdat[rawdat < 0] = 0


    
    write_output_file(glt_ds, colorized, args.output_file)
    plume_polygons = None
    if args.output_mask is not None:
        write_output_file(ds, plume, args.output_mask)
        outmask_ort_file = os.path.splitext(args.output_mask)[0] + '_ort.tif'
        outmask_poly_file = os.path.splitext(args.output_mask)[0] + '_polygon.json'
        write_output_file(glt_ds, plume_labels, outmask_ort_file)
        subprocess.call(f'rm {outmask_poly_file}',shell=True)
        subprocess.call(f'gdal_polygonize.py {outmask_ort_file} {outmask_poly_file} -f GeoJSON -mask {outmask_ort_file}',shell=True)
        raw_plume_polygons = json.load(open(outmask_poly_file))
        plume_polygons = {}
        for feat in raw_plume_polygons['features']:
            plume_polygons[str(feat['properties']['DN'])] = feat['geometry']
        print(plume_polygons) 


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


    proj_ds = gdal.Warp('', args.gltfile, dstSRS='EPSG:3857', format='VRT')
    transform_3857 = proj_ds.GetGeoTransform()
    xsize_m = transform_3857[1]
    ysize_m = transform_3857[5]
    del proj_ds
    print(xsize_m, ysize_m)


    # Make output vector
    outdict = {"crs": {"properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"}, "type": "name"},"features":[],"name":"methane_metadata","type":"FeatureCollection" }
    un_labels = np.unique(plume_labels)[1:]
    loclist = []
    for _lab, lab in enumerate(un_labels):
        
        maxval = np.nanmax(rawdat[plume_labels == lab])
        if np.sum(plume_labels == lab) < 15 or maxval < 200:
            continue

        rawloc = np.where(np.logical_and(rawdat == maxval, plume_labels == lab))
        maxval = np.round(maxval)

        #sum and convert to kg.  conversion:
        # ppm m / 1e6 ppm * x_pixel_size(m)*y_pixel_size(m) 1e3 L / m^3 * 1 mole / 22.4 L * 0.01604 kg / mole
        ime_scaler = (1.0/1e6)* ((np.abs(xsize_m*ysize_m))/1.0) * (1000.0/1.0) * (1.0/22.4)*(0.01604/1.0)

        ime_ss = rawdat[plume_labels == lab].copy()
        ime = np.nansum(ime_ss) * ime_scaler
        ime_p = np.nansum(ime_ss[ime_ss > 0]) * ime_scaler
        #if np.isfinite(ime) is False:
        #    print(rawdat[plume_labels == lab])
        #    print(np.nansum(rawdat[plume_labels == lab]))

        ime = np.round(ime,2)
        ime_uncert = np.round(np.sum(plume_labels == lab) * background * ime_scaler,2)


        y_locs = np.where(np.sum(plume_labels == lab, axis=1))[0]
        x_locs = np.where(np.sum(plume_labels == lab, axis=0))[0]
        radius = np.sqrt( ((y_locs[0] - y_locs[-1])*trans[5])**2 + ((x_locs[0] - x_locs[-1])*trans[1])**2)/2.
        center_y = trans[3] + trans[5] * (y_locs[0] + (y_locs[-1]-y_locs[0])/2)
        center_x = trans[0] + trans[1] * (x_locs[0] + (x_locs[-1]-x_locs[0])/2)

        point = shapely.geometry.Point(center_x, center_y)
        circle = shapely.geometry.polygon.Polygon(point.buffer(radius))


        #loclist.append(rawloc)
        max_loc_y = trans[3] + trans[5]*(rawloc[0][0]+0.5)
        max_loc_x = trans[0] + trans[1]*(rawloc[1][0]+0.5)


        if args.daac_demo_dir is not None:
            out_l3_name = os.path.join(args.daac_demo_dir, 'l3', os.path.splitext(os.path.basename(args.gltfile).replace('glt','ch4mf'))[0] + f'_mv0_p{_lab}.tif')
            outl3 = rawdat[y_locs[0]:y_locs[-1],x_locs[0]:x_locs[-1]].copy()
            outl3[np.isnan(outl3)] = -9999
            outtrans = list(glt_ds.GetGeoTransform()).copy()
            outtrans[0] += x_locs[0] * outtrans[1]
            outtrans[3] += y_locs[0] * outtrans[5]
            print(_lab, x_locs[0], y_locs[0])

            write_science_cog(outl3, out_l3_name, outtrans, glt_ds.GetProjection())



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
        #loc_res = {"geometry": {"coordinates": [center_x, center_y, 0.0], "type": "Point"},
        #           "vis_style": {"radius": radius},
        #           "type": "Feature",
        #           "properties": {"UTC Time Observed": start_datetime, 
        #                          "map_endtime": end_datetime,
        #                          "Max Plume Concentration (ppm m)": maxval,
        #                          "Concentration Uncertainty (ppm m)": background,
        #                          "Integrated Methane Enhancement (kg CH4)": ime,
        #                          "Integrated Methane Enhancement Uncertainty (kg CH4)": ime_uncert,
        #                          "Latitude of max concentration": max_loc_y,
        #                          "Longitude of max concentration": max_loc_x,
        #                          "Scene FID": fid,
        #                          "L1B Radiance Download": l1b_link
        #                          }}
        content_props = {"UTC Time Observed": start_datetime, 
                                      "map_endtime": end_datetime,
                                      "Max Plume Concentration (ppm m)": maxval,
                                      "Concentration Uncertainty (ppm m)": background,
                                      #"Integrated Methane Enhancement (kg CH4)": ime,
                                      #"Integrated Methane Enhancement - Positive (kg CH4)": ime_p,
                                      #"Integrated Methane Enhancement Uncertainty (kg CH4)": ime_uncert,
                                      "Latitude of max concentration": max_loc_y,
                                      "Longitude of max concentration": max_loc_x,
                                      "Scene FID": fid,
                                      "L1B Radiance Download": l1b_link
                         }
        if plume_polygons is not None:
            props = content_props.copy()
            props['style'] = {"maxZoom": 20, "minZoom": 0, "color": "white", 'opacity': 1, 'weight': 2, 'fillOpacity': 0}
            loc_res = {"geometry": plume_polygons[str(int(lab))],
                       "type": "Feature",
                       "properties": props}

            outdict['features'].append(loc_res)
        
        props = content_props.copy()
        props['style'] = {"radius": 10, 'minZoom': 0, "maxZoom": 9, "color": "red", 'opacity': 1, 'weight': 2, 'fillOpacity': 0}
        #loc_res = {"geometry": {"coordinates": [center_x, center_y, 0.0], "type": "Point"},
        loc_res = {"geometry": {"coordinates": [max_loc_x, max_loc_y, 0.0], "type": "Point"},
                   "properties": props,
                   "type": "Feature"}
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

