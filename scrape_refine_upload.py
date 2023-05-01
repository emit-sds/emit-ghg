import argparse

import os
import scipy
import numpy as np
import json
import glob
import datetime
import subprocess
from segment_anything import sam_model_registry, SamPredictor
from spectral.io import envi
import shapely
from shapely.geometry import Point
from shapely.geometry import Polygon
from copy import deepcopy
from osgeo import gdal
from apply_glt import single_image_ortho
from masked_plume_delineator import write_output_file
from skimage.draw import polygon
from rasterio.features import rasterize
from masked_plume_delineator import plume_mask
import logging


def roi_filter(coverage, roi):
    subdir = deepcopy(coverage)
    subdir['features'] = []
    target = Polygon(roi)
    for feat in coverage['features']:
        source = Polygon(feat['geometry']['coordinates'][0])
        if source.intersects(roi):
            subdir['features'].append(feat)
    return subdir
 
def time_filter(coverage, start_time, end_time):
    subdir = deepcopy(coverage)
    subdir['features'] = []
    for feat in coverage['features']:
      if start_time is None or datetime.datetime.strptime(feat['properties']['start_time'],'%Y-%m-%dT%H:%M:%SZ') >= datetime.datetime.strptime(start_time, '%Y-%m-%dT%H:%M:%S'):
        if end_time is None or datetime.datetime.strptime(feat['properties']['end_time'],'%Y-%m-%dT%H:%M:%SZ') <= datetime.datetime.strptime(end_time, '%Y-%m-%dT%H:%M:%S'):
          subdir['features'].append(feat)
    return subdir


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


def add_fids(manual_annotations, coverage, manual_annotations_previous):
    manual_annotations_fid = deepcopy(manual_annotations)

    previous_plume_ids = []
    if manual_annotations_previous is not None:
        previous_plume_ids = [x['properties']['Plume ID'] for x in manual_annotations_previous['features']]

    updated_plumes=[]
    for _feat, feat in enumerate(manual_annotations['features']):
        plume_id = feat['properties']['Plume ID']
        if plume_id in previous_plume_ids:
            new_geom = feat['geometry']['coordinates']
            prev_idx = previous_plume_ids.index(plume_id)
            prev_geom = manual_annotations_previous['features'][prev_idx]['geometry']['coordinates']
            if new_geom == prev_geom:
                manual_annotations_fid['features'][_feat] = manual_annotations_previous['features'][prev_idx]
                continue

        subset_coverage = time_filter(roi_filter(coverage, Polygon(feat['geometry']['coordinates'][0])), feat['properties']['Time Range Start'], feat['properties']['Time Range End'])
        fid = subset_coverage['features'][0]['properties']['fid'].split('_')[0]
        manual_annotations_fid['features'][_feat]['fid'] = fid
        updated_plumes.append(_feat)
    return manual_annotations_fid, updated_plumes




def main(input_args=None):
    parser = argparse.ArgumentParser(description="merge jsons")
    parser.add_argument('key', type=str,  metavar='INPUT_DIR', help='input directory')   
    parser.add_argument('id', type=str,  metavar='INPUT_DIR', help='input directory')   
    parser.add_argument('out_dir', type=str,  metavar='INPUT_DIR', help='input directory')   
    parser.add_argument('-style', type=str,  choices=['classic','sam'], default='classic')   
    parser.add_argument('--loglevel', type=str, default='DEBUG', help='logging verbosity')    
    parser.add_argument('--logfile', type=str, default=None, help='output file to write log to')    
    parser.add_argument('--continuous', action='store_true', help='run continuously')    
    args = parser.parse_args(input_args)

    logging.basicConfig(format='%(levelname)s:%(asctime)s ||| %(message)s', level=args.loglevel,
                        filename=args.logfile, datefmt='%Y-%m-%d,%H:%M:%S')

    np.random.seed(13)

    max_runs = 1
    if args.continuous:
        max_runs = int(1e15)

    for run in range(max_runs):
        logging.debug('Loading Data')
        previous_annotation_file = os.path.join(args.out_dir, "previous_manual_annotation.json")
        annotation_file = os.path.join(args.out_dir, "manual_annotation.json")
        subprocess.call(f'curl "https://popo.jpl.nasa.gov/mmgis/API/files/getfile" -H "Authorization:Bearer {args.key}" --data-raw "id={args.id}" > {annotation_file}',shell=True)
        manual_annotations = json.load(open(annotation_file,'r'))['body']['geojson']
        manual_annotations_previous = None
        if os.path.isfile(previous_annotation_file):
            manual_annotations_previous = json.load(open(previous_annotation_file,'r'))['body']['geojson']

        coverage = json.load(open('/beegfs/scratch/brodrick/emit/emit-visuals/track_coverage_pub.json','r'))

        logging.debug('Set up outputs')
        output_json = os.path.join(args.out_dir, 'combined_plume_metadata.json')
        outdict = {"crs": {"properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84" }, "type": "name"},"features":[],"name":"methane_metadata","type":"FeatureCollection" }
        if os.path.isfile(output_json):
            outdict = json.load(open(output_json,'r'))

        # Preload SAM model if needed
        if args.style == 'sam':
            sam_checkpoint = "sam_vit_h_4b8939.pth"
            model_type = "vit_h"
            sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
            #sam.to(device=device)
            predictor = SamPredictor(sam)


        manual_annotations, new_plumes = add_fids(manual_annotations, coverage, manual_annotations_previous)
        print(new_plumes)
        new_plume_fids = [manual_annotations['features'][x]['fid'] for x in new_plumes]

        if len(new_plumes) > 0:
            unique_fids = np.unique(new_plume_fids)

            for fid in unique_fids:

                new_plumes_in_fid = [x  for x in new_plumes if new_plume_fids[x] == fid]
                this_fid_manual_annotations = deepcopy(manual_annotations)
                this_fid_manual_annotations['features'] = [this_fid_manual_annotations['features'][x] for x in new_plumes_in_fid]

                rawdat = envi.open(f'methane_20221121/{fid.split("_")[0]}_ch4_mf.hdr').open_memmap(interleave='bip').copy()
                ds = gdal.Open(f'methane_20221121/{fid.split("_")[0]}_ch4_mf')


                image = rawdat.copy()
                rawdat = rawdat.squeeze()
                rawdat[rawdat == ds.GetRasterBand(1).GetNoDataValue()] = np.nan


                if args.style == 'sam':
                    image = (image / 1500).astype(np.uint8)
                    image[image > 1] = 1
                    image[image < 0] = 0
                    image *= 255
                    oi = np.zeros((image.shape[0],image.shape[1],3),dtype=np.uint8)
                    oi[...] = image.squeeze()[:,:,np.newaxis]
                    image = oi
                    del oi

                    predictor.set_image(image)
              
                # Load GLT
                glt_file = glob.glob(f'/beegfs/store/emit/ops/data/acquisitions/{fid[4:12]}/{fid.split("_")[0]}/l1b/*_glt_b0106_v01.img')[0]
                glt_ds = gdal.Open(glt_file)
                glt = glt_ds.ReadAsArray().transpose((1,2,0))
                trans = glt_ds.GetGeoTransform()

                rawshape = (rawdat.shape[0],rawdat.shape[1])


                full_fid_mask = np.zeros(rawshape)
                plume_labels = np.zeros(rawshape)
                for newp in new_plumes_in_fid:

                    feat = manual_annotations['features'][newp]

                    rawspace_coords = []
                    for ind in feat['geometry']['coordinates'][0]:
                        glt_ypx = int(round((ind[1] - trans[3])/ trans[5]))
                        glt_xpx = int(round((ind[0] - trans[0])/ trans[1]))
                        lglt = glt[glt_ypx, glt_xpx,:]
                        rawspace_coords.append(lglt.tolist())

                    manual_mask = rasterize(shapes=[Polygon(rawspace_coords)], out_shape=rawshape)

                    # Do segmenation
                    if args.style == 'sam':
                        min_x = np.min([x[0] for x in rawspace_coords])
                        max_x = np.max([x[0] for x in rawspace_coords])
                        min_y = np.min([x[1] for x in rawspace_coords])
                        max_y = np.max([x[1] for x in rawspace_coords])
                        bbox = np.array([min_x, min_y, max_x, max_y])

                        n_input_points = 20
                        input_labels, input_points = [],[]
                        for _n in range(n_input_points):
                            pt = [np.random.randint(min_x, max_x), np.random.randint(min_y,max_y)]
                            input_labels.append(manual_mask[pt[1],pt[0]])
                            input_points.append(pt)
                            
                        print(input_points)
                        print(input_labels)
                        masks, _, _ = predictor.predict(
                                                        point_coords=np.array(input_points),
                                                        point_labels=np.array(input_labels),
                                                        box=bbox,
                                                        multimask_output=False,
                                                       )
                        full_fid_mask += masks[0,...]
                        plume_labels[masks[0,...] == 1] = newp
                        #plume_labels = single_image_ortho(np.transpose(masks, (1,2,0)), glt)[...,0]

                    elif args.style == 'classic':
                        full_fid_mask += manual_mask

                
                if args.style == 'classic':
                    #seg_dat = rawdat.copy()
                    #seg_dat[np.logical_not(plume)] = np.nan
                    #seg_dat = gauss_blur(dat, args.blur_sigma)   
                    #seg_dat[np.logical_not(plume)] = 0

                    #seg_dat /= 1500
                    #seg_dat[dat > 1] = 1
                    #seg_dat[dat < 0] = 0
                    #seg_dat[dat == 0] = 0.01
        
                    full_fid_mask, plume_labels = plume_mask(rawdat, manual_mask)
                

                # ortho
                plume_labels = single_image_ortho(plume_labels.reshape((rawshape[0],rawshape[1],1)), glt)[...,0]
                plume_labels[plume_labels == -9999] = 0

                outmask_ort_file = os.path.join(args.out_dir, f'{fid}_ort.tif')
                outmask_poly_file = os.path.join(args.out_dir, f'{fid}_polygon.json')
                write_output_file(glt_ds, plume_labels, outmask_ort_file)
                subprocess.call(f'rm {outmask_poly_file}',shell=True)
                subprocess.call(f'gdal_polygonize.py {outmask_ort_file} {outmask_poly_file} -f GeoJSON -mask {outmask_ort_file}',shell=True)

                raw_plume_polygons = json.load(open(outmask_poly_file))
                plume_polygons = {}
                for poly_feat in raw_plume_polygons['features']:

                    manual_matches = roi_filter(this_fid_manual_annotations, Polygon(poly_feat['geometry']['coordinates'][0]))
                    if len(manual_matches['features']) > 1:
                        logging.warn('ACK - We intersected against multiple plumes')
                    if len(manual_matches['features']) == 0:
                        logging.warn('ACK - We couldnt find an intersection')

                    print(manual_matches['features'][0]['properties'])
                    plume_polygons[str(poly_feat['properties']['DN'])] = {'geometry': poly_feat['geometry'], 'Plume ID': manual_matches['features'][0]['properties']['Plume ID'], 'Reviewer 1 Approval': manual_matches['features'][0]['properties']['Reviewer 1 Approval'] }
                    # add metadata from the mathed plume



                # Get Radiance link
                l1b_rad = glob.glob(f'/beegfs/store/emit/ops/data/acquisitions/{fid[4:12]}/{fid.split("_")[0]}/l1b/EMIT_L1B_RAD*_cnm.out')
                if len(l1b_rad) > 0:
                    l1b_rad = os.path.basename(l1b_rad[0]).split('.')[0][:-20]
                    l1b_link = f'https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL1BRAD.001/{l1b_rad}/{l1b_rad}.nc'
                else:
                    l1b_link = f'Scene Not Yet Available'

                proj_ds = gdal.Warp('', glt_file, dstSRS='EPSG:3857', format='VRT')
                transform_3857 = proj_ds.GetGeoTransform()
                xsize_m = transform_3857[1]
                ysize_m = transform_3857[5]
                del proj_ds

####    ############
                rawdat = single_image_ortho(rawdat.reshape((rawshape[0], rawshape[1], 1)), glt)[...,0]
                rawdat[rawdat == -9999] = np.nan

                un_labels = np.unique(plume_labels)[1:]
                background = np.round(np.nanstd(rawdat[plume_labels == 0]))
                loclist = []
                start_datetime = datetime.datetime.strptime(fid[4:], "%Y%m%dt%H%M%S")
                end_datetime = start_datetime + datetime.timedelta(seconds=1)

                start_datetime = start_datetime.strftime("%Y-%m-%dT%H:%M:%SZ")
                end_datetime = end_datetime.strftime("%Y-%m-%dT%H:%M:%SZ")


                for lab in un_labels:
                    
                    maxval = np.nanmax(rawdat[plume_labels == lab])
                    if np.sum(plume_labels == lab) < 5 or maxval < 200:
                        continue

                    rawloc = np.where(np.logical_and(rawdat == maxval, plume_labels == lab))
                    maxval = np.round(maxval)

                    #sum and convert to kg.  conversion:
                    # ppm m / 1e6 ppm * x_pixel_size(m)*y_pixel_size(m) 1e3 L / m^3 * 1 mole / 22.4 L * 0.01604 kg / mole
                    ime_scaler = (1.0/1e6)* ((np.abs(xsize_m*ysize_m))/1.0) * (1000.0/1.0) * (1.0/22.4)*(0.01604/1.0)

                    ime_ss = rawdat[plume_labels == lab].copy()
                    ime = np.nansum(ime_ss) * ime_scaler
                    ime_p = np.nansum(ime_ss[ime_ss > 0]) * ime_scaler

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
                    #loc_z = 0

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
                        loc_pp = plume_polygons[str(int(lab))]
                        props['Plume ID'] = loc_pp['Plume ID']

                        
                        if loc_pp['Reviewer 1 Approval']:
                            props['style'] = {"maxZoom": 20, "minZoom": 0, "color": "white", 'opacity': 1, 'weight': 2, 'fillOpacity': 0}
                        else:
                            props['style'] = {"maxZoom": 20, "minZoom": 0, "color": "green", 'opacity': 1, 'weight': 2, 'fillOpacity': 0}
                        loc_res = {"geometry": loc_pp['geometry'],
                                   "type": "Feature",
                                   "properties": props}

                        existing_match_index = [_x for _x, x in enumerate(outdict['features']) if props['Plume ID'] == x['properties']['Plume ID'] and x['geometry']['type'] != 'Point']
                        if len(existing_match_index) > 2:
                            logging.warn("HELP! Too many matching indices")
                        if len(existing_match_index) > 0:
                            outdict['features'][existing_match_index[0]] = loc_res
                        else:
                            outdict['features'].append(loc_res)
                    
                        props = content_props.copy()
                        props['style'] = {"radius": 10, 'minZoom': 0, "maxZoom": 9, "color": "red", 'opacity': 1, 'weight': 2, 'fillOpacity': 0}
                        props['Plume ID'] = loc_pp['Plume ID']
                        #loc_res = {"geometry": {"coordinates": [center_x, center_y, 0.0], "type": "Point"},
                        loc_res = {"geometry": {"coordinates": [max_loc_x, max_loc_y, 0.0], "type": "Point"},
                                   "properties": props,
                                   "type": "Feature"}
                        outdict['features'].append(loc_res)
                        existing_match_index = [_x for _x, x in enumerate(outdict['features']) if props['Plume ID'] == x['properties']['Plume ID'] and x['geometry']['type'] == 'Point']
                        if len(existing_match_index) > 2:
                            logging.warn("HELP! Too many matching indices")
                        if len(existing_match_index) > 0:
                            outdict['features'][existing_match_index[0]] = loc_res
                        else:
                            outdict['features'].append(loc_res)

            outdict['crs']['properties']['last_updated'] = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            with open(output_json, 'w') as fout:
                fout.write(json.dumps(outdict, cls=SerialEncoder, sort_keys=True)) 
            subprocess.call(f'cp {previous_annotation_file} {os.path.splitext(previous_annotation_file)[0] + "_oneback.json"}',shell=True)
            subprocess.call(f'cp {annotation_file} {previous_annotation_file}',shell=True)
            subprocess.call(f'rsync {output_json} brodrick@$EMIT_SCIENCE_IP:/data/emit/mmgis/coverage/converted_manual_plumes.json',shell=True)

 
if __name__ == '__main__':
    main()


