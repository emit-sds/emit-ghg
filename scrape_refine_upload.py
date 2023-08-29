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
import os
import numpy as np
import json
import glob
import datetime
import time
import subprocess
from spectral.io import envi
from shapely.geometry import Polygon
from copy import deepcopy
from osgeo import gdal
from apply_glt import single_image_ortho
from masked_plume_delineator import write_output_file
from rasterio.features import rasterize
import logging
import matplotlib.pyplot as plt
from masked_plume_delineator import gauss_blur



def plume_mask_threshold(input: np.array, pm, style='ch4'):

    y_locs = np.where(np.sum(pm > 0, axis=1))[0]
    x_locs = np.where(np.sum(pm > 0, axis=0))[0]

    plume_dat = input[y_locs[0]:y_locs[-1],x_locs[0]:x_locs[-1]].copy()
    plume_dat[pm[y_locs[0]:y_locs[-1],x_locs[0]:x_locs[-1]] == 0] = 0
    
    local_output_mask = None
    if style == 'ch4':
        plume_dat = gauss_blur(plume_dat, 3, preserve_nans=False)
        local_output_mask = plume_dat > 50
    else:
        plume_dat = gauss_blur(plume_dat, 5, preserve_nans=False)
        local_output_mask = plume_dat > 10000
    
    output_plume_mask = np.zeros(pm.shape,dtype=bool)
    output_plume_mask[y_locs[0]:y_locs[-1],x_locs[0]:x_locs[-1]] = local_output_mask
    
    return output_plume_mask



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
    todel=[]
    for _feat, feat in enumerate(manual_annotations['features']):
        # If this key isn't present, then the full feature wasn't really added yet
        if 'R1 - Reviewed' not in feat['properties'].keys():
            todel.append(_feat)
            continue # This is insufficient
        plume_id = feat['properties']['Plume ID']
        if plume_id in previous_plume_ids:
            new_geom = feat['geometry']['coordinates']
            prev_idx = previous_plume_ids.index(plume_id)
            prev_geom = manual_annotations_previous['features'][prev_idx]['geometry']['coordinates']

            # check reviews
            rev_match = True
            for rl in ['R1 - Reviewed','R2 - Reviewed','R1 - VISIONS','R2 - VISIONS']:
                if feat['properties'][rl] != manual_annotations_previous['features'][prev_idx]['properties'][rl]:
                    rev_match = False

            if new_geom == prev_geom and rev_match:
                manual_annotations_fid['features'][_feat] = manual_annotations_previous['features'][prev_idx]
                continue

        subset_coverage = roi_filter(time_filter(coverage, feat['properties']['Time Range Start'], feat['properties']['Time Range End']), Polygon(feat['geometry']['coordinates'][0]))
        if len(subset_coverage['features']) == 0:
            todel.append(_feat)
        else:
            fid = subset_coverage['features'][0]['properties']['fid'].split('_')[0]
            manual_annotations_fid['features'][_feat]['properties']['fid'] = fid
            updated_plumes.append(_feat)

    for td in np.array(todel)[::-1]:
        msg = f'Deleting entry due to bad metadata - check input {manual_annotations_fid["features"][td]}'
        logging.warn(msg)
        manual_annotations_fid['features'].pop(td)

    updated_plumes = np.array([x for x in updated_plumes if x not in todel])
    for td in todel:
        updated_plumes[updated_plumes > td] -= 1
    updated_plumes = updated_plumes.tolist()

    return manual_annotations_fid, updated_plumes

def write_color_plume(rawdat, plumes_mask, glt_ds, outname: str, style = 'ch4'):

    dat = rawdat.copy()
    #dat = single_image_ortho(dat.reshape(dat.shape[0],dat.shape[1],1), glt).squeeze()
    colorized = np.zeros((dat.shape[0],dat.shape[1],3))

    dat[np.logical_not(plumes_mask)] = 0

    if style == 'ch4':
        dat /= 1500
        dat[dat > 1] = 1
        dat[dat < 0] = 0
        dat[dat == 0] = 0.01
        colorized[plumes_mask,:] = plt.cm.plasma(dat[plumes_mask])[...,:3]
    else:
        #dat /= 85000
        dat /= 85000
        dat[dat > 1] = 1
        dat[dat < 0] = 0
        dat[dat == 0] = 0.01
        colorized[plumes_mask,:] = plt.cm.viridis(dat[plumes_mask])[...,:3]


    colorized = np.round(colorized * 255).astype(np.uint8)
    colorized[plumes_mask,:] = np.maximum(1, colorized[plumes_mask,:])

    write_output_file(glt_ds, colorized, outname)


def prep_predictor_image(predictor, data, ptype):
    image = data.copy()
    if ptype == 'co2':
        image = (image / 1500).astype(np.uint8)
    else:
        image = (image / 100000).astype(np.uint8)
    image[image > 1] = 1
    image[image < 0] = 0
    image *= 255
    oi = np.zeros((image.shape[0],image.shape[1],3),dtype=np.uint8)
    oi[...] = image.squeeze()[:,:,np.newaxis]

    predictor.set_image(oi)

def rawspace_coordinate_conversion(glt, coordinates, trans, ortho=False):
    rawspace_coords = []
    for ind in coordinates:
        glt_ypx = int(round((ind[1] - trans[3])/ trans[5]))
        glt_xpx = int(round((ind[0] - trans[0])/ trans[1]))
        if ortho:
            rawspace_coords.append([glt_xpx,glt_ypx])
        else:
            lglt = glt[glt_ypx, glt_xpx,:]
            rawspace_coords.append(lglt.tolist())
    return rawspace_coords


def sam_segmentation(predictor, rawspace_coords, manual_mask, n_input_points=20):
    min_x = np.min([x[0] for x in rawspace_coords])
    max_x = np.max([x[0] for x in rawspace_coords])
    min_y = np.min([x[1] for x in rawspace_coords])
    max_y = np.max([x[1] for x in rawspace_coords])
    bbox = np.array([min_x, min_y, max_x, max_y])

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

    return masks[0,...]


def get_radiance_link(fid):
    # Get Radiance link
    l1b_rad = glob.glob(f'/beegfs/store/emit/ops/data/acquisitions/{fid[4:12]}/{fid.split("_")[0]}/l1b/EMIT_L1B_RAD*_cnm.out')
    if len(l1b_rad) > 0:
        l1b_rad = os.path.basename(l1b_rad[0]).split('.')[0][:-20]
        l1b_link = f'https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL1BRAD.001/{l1b_rad}/{l1b_rad}.nc'
    else:
        l1b_link = f'Scene Not Yet Available'
    return l1b_link
          


def main(input_args=None):
    parser = argparse.ArgumentParser(description="merge jsons")
    parser.add_argument('key', type=str,  metavar='INPUT_DIR', help='input directory')   
    parser.add_argument('id', type=str,  metavar='INPUT_DIR', help='input directory')   
    parser.add_argument('out_dir', type=str,  metavar='INPUT_DIR', help='input directory')   
    parser.add_argument('--type', type=str,  choices=['ch4','co2'], default='ch4')   
    parser.add_argument('--loglevel', type=str, default='DEBUG', help='logging verbosity')    
    parser.add_argument('--logfile', type=str, default=None, help='output file to write log to')    
    parser.add_argument('--continuous', action='store_true', help='run continuously')    
    parser.add_argument('--track_coverage_file', default='/beegfs/scratch/brodrick/emit/emit-visuals/track_coverage_pub.json')
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
        subprocess.call(f'/beegfs/scratch/brodrick/miniconda/envs/isofit_env/bin/curl "https://popo.jpl.nasa.gov/mmgis/API/files/getfile" -H "Authorization:Bearer {args.key}" --data-raw "id={args.id}" > {annotation_file}',shell=True)
        manual_annotations = json.load(open(annotation_file,'r'))['body']['geojson']
        for _feat in range(len(manual_annotations['features'])):
            manual_annotations['features'][_feat]['properties']['Plume ID'] = manual_annotations['features'][_feat]['properties'].pop('name')


        manual_annotations_previous = None
        if os.path.isfile(previous_annotation_file):
            manual_annotations_previous = json.load(open(previous_annotation_file,'r'))

        coverage = json.load(open(args.track_coverage_file,'r'))

        logging.debug('Set up outputs')
        output_json = os.path.join(args.out_dir, 'combined_plume_metadata.json')
        outdict = {"crs": {"properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84" }, "type": "name"},"features":[],"name":"methane_metadata","type":"FeatureCollection" }
        if os.path.isfile(output_json):
            outdict = json.load(open(output_json,'r'))


        # Step through each new plume
        manual_annotations, new_plumes = add_fids(manual_annotations, coverage, manual_annotations_previous)
        # Dump out the udpated manual annotations set, so it holds FIDs for next round
        with open(annotation_file, 'w') as fout:
            fout.write(json.dumps(manual_annotations, cls=SerialEncoder)) 

        new_plume_fids = np.unique([manual_annotations['features'][x]['properties']['fid'] for x in new_plumes]).tolist()

        # If there's nothing new, sleep and retry
        if len(new_plumes) == 0:
            time.sleep(10)
            continue

        # Narrow down the new plumes to the set of unique FIDS, so we step through scene by scene
        unique_fids = np.unique(new_plume_fids)
        for fid in unique_fids:

            # Get just the plumes that are in this FID
            #new_plumes_in_fid = [x for x in new_plumes if manual_annotations['features'][x]['fid'] == fid]
            plumes_in_fid = [x for x in range(len(manual_annotations['features'])) if manual_annotations['features'][x]['properties']['fid'] == fid]
            
            this_fid_manual_annotations = deepcopy(manual_annotations)
            #this_fid_manual_annotations['features'] = [this_fid_manual_annotations['features'][x] for x in new_plumes_in_fid]
            this_fid_manual_annotations['features'] = [this_fid_manual_annotations['features'][x] for x in plumes_in_fid]

            # Open up the relevant MF data
            rawdat = envi.open(f'methane_20221121/{fid.split("_")[0]}_{args.type}_mf.hdr').open_memmap(interleave='bip').copy()
            ds = gdal.Open(f'methane_20221121/{fid.split("_")[0]}_{args.type}_mf')

            rawdat = rawdat.squeeze()
            rawdat[rawdat == ds.GetRasterBand(1).GetNoDataValue()] = np.nan
            rawshape = (rawdat.shape[0],rawdat.shape[1])

            # Load GLT
            glt_file = glob.glob(f'/beegfs/store/emit/ops/data/acquisitions/{fid[4:12]}/{fid.split("_")[0]}/l1b/*_glt_b0106_v01.img')[0]
            glt_ds = gdal.Open(glt_file)
            glt = glt_ds.ReadAsArray().transpose((1,2,0))
            trans = glt_ds.GetGeoTransform()

            # ortho rawdat
            ort_rawdat = single_image_ortho(rawdat.reshape((rawshape[0], rawshape[1], 1)), glt)[...,0]
            ort_rawdat[ort_rawdat == -9999] = np.nan

            # Use the manual plumes to come up with a new set of plume masks and labels
            #for newp in new_plumes_in_fid:
            for newp in plumes_in_fid:
                feat = manual_annotations['features'][newp]
                #rawspace_coords = rawspace_coordinate_conversion(glt, feat['geometry']['coordinates'][0], trans)
                #manual_mask = rasterize(shapes=[Polygon(rawspace_coords)], out_shape=rawshape) # numpy binary mask for manual IDs
                rawspace_coords = rawspace_coordinate_conversion(glt, feat['geometry']['coordinates'][0], trans, ortho=True)
                manual_mask = rasterize(shapes=[Polygon(rawspace_coords)], out_shape=(glt.shape[0],glt.shape[1])) # numpy binary mask for manual IDs

                plumestyle = 'classic'
                if 'Delineation Mode' in feat['properties'].keys():
                    plumestyle = feat['properties']['Delineation Mode']

                loc_fid_mask = None
                if plumestyle == 'classic':
                    #loc_fid_mask = plume_mask_threshold(rawdat.copy(), manual_mask, style=args.type)
                    loc_fid_mask = plume_mask_threshold(ort_rawdat.copy(), manual_mask, style=args.type)
                elif plumestyle == 'manual':    
                    loc_fid_mask = manual_mask.astype(bool)

                outmask_poly_file = os.path.join(args.out_dir, f'{fid}_{feat["properties"]["Plume ID"]}_polygon.json')
                outmask_ort_file = os.path.join(args.out_dir, f'{fid}_{feat["properties"]["Plume ID"]}_mask_ort.tif')
                write_output_file(glt_ds, loc_fid_mask, outmask_ort_file)
                subprocess.call(f'rm {outmask_poly_file}',shell=True)
                subprocess.call(f'gdal_polygonize.py {outmask_ort_file} {outmask_poly_file} -f GeoJSON -mask {outmask_ort_file} -8',shell=True)


            # Now collect everything from the FID....new and old
            full_fid_mask = np.zeros((glt.shape[0],glt.shape[1]),dtype=bool) # mask of where plumes are in fid
            plume_labels = np.zeros((glt.shape[0],glt.shape[1]),dtype=int) # labels of individual plumes in fid

            outmask_poly_files = glob.glob(os.path.join(args.out_dir, f'{fid}_*_polygon.json'))
            outmask_ort_files = [x.replace("_polygon.json","_mask_ort.tif") for x in outmask_poly_files]
            plume_polygons = {}
            for _fid_poly_file in range(len(outmask_poly_files)):
                
                plume_to_add = json.load(open(outmask_poly_files[_fid_poly_file]))['features']
                if len(plume_to_add) > 1:
                    logging.warn('ACK - multiple polygons from one Plume ID')
                plume_to_add = plume_to_add[0]

                loc_fid_mask = np.squeeze(gdal.Open(outmask_ort_files[_fid_poly_file]).ReadAsArray()).astype(bool)
                full_fid_mask[loc_fid_mask] = 1
                plume_labels[loc_fid_mask] = _fid_poly_file+1

                plume_id = os.path.basename(outmask_poly_files[_fid_poly_file])[20:].replace('_polygon.json','') 
                match_idx = np.unique([x for x, matchfeat in enumerate(this_fid_manual_annotations['features']) if matchfeat['properties']['Plume ID'] == plume_id])
                if len(match_idx) > 1:
                    logging.warn('ACK - We intersected against multiple plumes')
                    continue
                if len(match_idx) == 0:
                    logging.warn('ACK - We couldnt find an intersection')
                    continue
                match_idx = match_idx[0]

                plume_polygons[str(_fid_poly_file + 1)] = this_fid_manual_annotations['features'][match_idx]['properties']
                plume_polygons[str(_fid_poly_file + 1)]['geometry'] = plume_to_add['geometry']

            color_ort_file = os.path.join(args.out_dir, f'{fid}_color_ort.tif')
            write_color_plume(ort_rawdat, full_fid_mask, glt_ds, color_ort_file, style=args.type)

            #plume_polygons = {}
            #for poly_feat in raw_plume_polygons['features']:

            #    manual_matches = roi_filter(this_fid_manual_annotations, Polygon(poly_feat['geometry']['coordinates'][0]))
            #    if len(manual_matches['features']) > 1:
            #        logging.warn('ACK - We intersected against multiple plumes')
            #        continue
            #    if len(manual_matches['features']) == 0:
            #        logging.warn('ACK - We couldnt find an intersection')
            #        continue


            #    logging.info(f'New plumes: {manual_matches["features"][0]["properties"]}')
            #    plume_polygons[str(poly_feat['properties']['DN'])] = {'geometry': poly_feat['geometry']}
            #    for ppk,ppi in manual_matches['features'][0]['properties'].items():
            #        plume_polygons[str(poly_feat['properties']['DN'])][ppk] = ppi


            l1b_link = get_radiance_link(fid)

            proj_ds = gdal.Warp('', glt_file, dstSRS='EPSG:3857', format='VRT')
            transform_3857 = proj_ds.GetGeoTransform()
            xsize_m = transform_3857[1]
            ysize_m = transform_3857[5]
            del proj_ds

            un_labels = np.unique(plume_labels)[1:]
            background = np.round(np.nanstd(ort_rawdat[plume_labels == 0]))
            start_datetime = datetime.datetime.strptime(fid[4:], "%Y%m%dt%H%M%S")
            end_datetime = start_datetime + datetime.timedelta(seconds=1)

            start_datetime = start_datetime.strftime("%Y-%m-%dT%H:%M:%SZ")
            end_datetime = end_datetime.strftime("%Y-%m-%dT%H:%M:%SZ")


            for lab in un_labels:
                
                maxval = np.nanmax(ort_rawdat[plume_labels == lab])
                if np.sum(plume_labels == lab) < 5 or maxval < 200 or np.isnan(maxval):
                    continue

                rawloc = np.where(np.logical_and(ort_rawdat == maxval, plume_labels == lab))
                maxval = np.round(maxval)

                #sum and convert to kg.  conversion:
                # ppm m / 1e6 ppm * x_pixel_size(m)*y_pixel_size(m) 1e3 L / m^3 * 1 mole / 22.4 L * 0.01604 kg / mole
                ime_scaler = (1.0/1e6)* ((np.abs(xsize_m*ysize_m))/1.0) * (1000.0/1.0) * (1.0/22.4)*(0.01604/1.0)

                ime_ss = ort_rawdat[plume_labels == lab].copy()
                ime = np.nansum(ime_ss) * ime_scaler
                ime_p = np.nansum(ime_ss[ime_ss > 0]) * ime_scaler

                ime = np.round(ime,2)
                ime_uncert = np.round(np.sum(plume_labels == lab) * background * ime_scaler,2)

                max_loc_y = trans[3] + trans[5]*(rawloc[0][0]+0.5)
                max_loc_x = trans[0] + trans[1]*(rawloc[1][0]+0.5)

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

                    # For R1 Review
                    if not loc_pp['R1 - Reviewed']:
                        props['style'] = {"maxZoom": 20, "minZoom": 0, "color": "red", 'opacity': 1, 'weight': 2, 'fillOpacity': 0}
                    
                    # For R2 Review
                    if loc_pp['R1 - Reviewed'] and loc_pp['R1 - VISIONS'] and not loc_pp['R2 - Reviewed']:
                        props['style'] = {"maxZoom": 20, "minZoom": 0, "color": "green", 'opacity': 1, 'weight': 2, 'fillOpacity': 0}

                    # Accept
                    if loc_pp['R1 - Reviewed'] and loc_pp['R1 - VISIONS'] and loc_pp['R2 - Reviewed'] and loc_pp['R2 - VISIONS']:
                        props['style'] = {"maxZoom": 20, "minZoom": 0, "color": "white", 'opacity': 1, 'weight': 2, 'fillOpacity': 0}
                    
                    # Reject
                    if (loc_pp['R1 - Reviewed'] and not loc_pp['R1 - VISIONS']) or (loc_pp['R2 - Reviewed'] and not loc_pp['R2 - VISIONS']):
                        props['style'] = {"maxZoom": 20, "minZoom": 0, "color": "yellow", 'opacity': 1, 'weight': 2, 'fillOpacity': 0}

                    loc_res = {"geometry": loc_pp['geometry'],
                               "type": "Feature",
                               "properties": props}

                    props['style']['radius'] = 10
                    point_res = {"geometry": {"coordinates": [max_loc_x, max_loc_y, 0.0], "type": "Point"},
                               "properties": props,
                               "type": "Feature"}

                    # First add in polygon
                    existing_match_index = [_x for _x, x in enumerate(outdict['features']) if props['Plume ID'] == x['properties']['Plume ID'] and x['geometry']['type'] != 'Point']
                    if len(existing_match_index) > 2:
                        logging.warn("HELP! Too many matching indices")
                    if len(existing_match_index) > 0:
                        outdict['features'][existing_match_index[0]] = loc_res
                    else:
                        outdict['features'].append(loc_res)
                
                    # Now add in point
                    existing_match_index = [_x for _x, x in enumerate(outdict['features']) if props['Plume ID'] == x['properties']['Plume ID'] and x['geometry']['type'] == 'Point']
                    if len(existing_match_index) > 2:
                        logging.warn("HELP! Too many matching indices")
                    if len(existing_match_index) > 0:
                        outdict['features'][existing_match_index[0]] = point_res
                    else:
                        outdict['features'].append(point_res)

            # Write JSON to output file
            outdict['crs']['properties']['last_updated'] = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            with open(output_json, 'w') as fout:
                fout.write(json.dumps(outdict, cls=SerialEncoder, sort_keys=True)) 

            # Tile and Sync
            date=fid[4:]
            ftime=fid.split('t')[-1].split('_')[0]
            tile_dir = os.path.join(args.out_dir, 'tiled_' + args.type)
            if os.path.isdir(tile_dir) is False:
                os.mkdir(tile_dir)
            od_date = f'{date[:4]}-{date[4:6]}-{date[6:8]}T{ftime[:2]}_{ftime[2:4]}_{ftime[4:]}Z-to-{date[:4]}-{date[4:6]}-{date[6:8]}T{ftime[:2]}_{ftime[2:4]}_{str(int(ftime[4:6])+1):02}Z'

            if os.path.isdir(f'{tile_dir}/{od_date}'):
                subprocess.call(f'rm -r {tile_dir}/{od_date}',shell=True)
            cmd_str = f'gdal2tiles.py -z 2-12 --srcnodata 0 --processes=40 -r antialias {color_ort_file} {tile_dir}/{od_date} -x'
            subprocess.call(cmd_str, shell=True)
            subprocess.call(f'rsync -a --info=progress2 {tile_dir}/{od_date}/ brodrick@$EMIT_SCIENCE_IP:/data/emit/mmgis/mosaics/{args.type}_plume_tiles_working/{od_date}/ --delete',shell=True)

            subprocess.call(f'cp {previous_annotation_file} {os.path.splitext(previous_annotation_file)[0] + "_oneback.json"}',shell=True)
            subprocess.call(f'cp {annotation_file} {previous_annotation_file}',shell=True)
            subprocess.call(f'rsync {output_json} brodrick@$EMIT_SCIENCE_IP:/data/emit/mmgis/coverage/converted_manual_{args.type}_plumes.json',shell=True)

 
if __name__ == '__main__':
    main()


