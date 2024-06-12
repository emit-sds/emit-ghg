import argparse
import subprocess
import datetime

import masked_plume_delineator
import logging
from spectral.io import envi
import numpy as np
import os
from utils import envi_header
from osgeo import gdal
import pandas as pd
import time
import json
import glob
from scrape_refine_upload import write_color_plume, rawspace_coordinate_conversion
from apply_glt import single_image_ortho
from copy import deepcopy
from rasterio.features import rasterize
from shapely.geometry import Polygon
import matplotlib.pyplot as plt
import requests

if os.environ.get("GHG_DEBUG"):
    logging.info("Using internal ray")
    import rray as ray
else:
    import ray




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

def write_science_cog(output_img, output_file, geotransform, projection, metadata):
    tmp_file = os.path.splitext(output_file)[0] + '_tmp.tif'
    driver = gdal.GetDriverByName('GTiff')
    driver.Register()
    outDataset = driver.Create(tmp_file,output_img.shape[1],output_img.shape[0],1,gdal.GDT_Float32,options = ['COMPRESS=LZW'])
    md = outDataset.GetMetadata()
    md.update(metadata)
    outDataset.SetMetadata(md)
    outDataset.GetRasterBand(1).WriteArray(output_img)
    outDataset.GetRasterBand(1).SetNoDataValue(-9999)
    outDataset.SetProjection(projection)
    outDataset.SetGeoTransform(geotransform)
    del outDataset

    subprocess.call(f'sh /home/brodrick/bin/cog.sh {tmp_file} {output_file} 1',shell=True)
    subprocess.call(f'rm {tmp_file}',shell=True)


def tile_dcid(features, outdir, datadir, gstyle='ch4'):

    dcid = features[0]["properties"]["DCID"]
    ds = gdal.Open(os.path.join(datadir, f'dcid_{dcid}_mf_ort.tif'))
    dat = ds.ReadAsArray().squeeze()

    plume_mask = np.zeros(dat.shape,dtype=bool)
    for feat in features:
       outmask_ort_file = os.path.join(datadir, f'{feat["properties"]["Plume ID"]}_mask_ort.tif')
       loc_dcid_mask = np.squeeze(gdal.Open(outmask_ort_file).ReadAsArray()).astype(bool)
       plume_mask[loc_dcid_mask] = 1

    color_ort_file = os.path.join(outdir, f'{dcid}_color_ort.tif')
    write_color_plume(dat, plume_mask, ds, color_ort_file, style=gstyle)

    fids = np.unique([sublist for feat in features for sublist in feat['properties']['Scene FIDs']])
    start_date=fids[0][4:]
    start_ftime=fids[0].split('t')[-1].split('_')[0]
    end_date=fids[-1][4:]
    end_ftime=fids[-1].split('t')[-1].split('_')[0]
    od_date = f'{start_date[:4]}-{start_date[4:6]}-{start_date[6:8]}T{start_ftime[:2]}_{start_ftime[2:4]}_{start_ftime[4:]}Z-to-{end_date[:4]}-{end_date[4:6]}-{end_date[6:8]}T{end_ftime[:2]}_{end_ftime[2:4]}_{str(int(end_ftime[4:6])+1):02}Z'
    cmd_str = f'gdal2tiles.py -z 2-12 --srcnodata 0 --processes=40 -r antialias {color_ort_file} {outdir}/{od_date} -x'
    subprocess.call(cmd_str,shell=True)
    

@ray.remote
def single_plume_proc(all_plume_meta, index, output_base, dcid_sourcedir, source_dir, extra_metadata, gtype='ch4'):

        kn = 'methane'
        if gtype == 'co2':
            kn = 'carbon_dioxide'
        plume_dict = {"crs": {"properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84" }, "type": "name"},"features":[],"name":f"{kn}_metadata","type":"FeatureCollection" }
        plume_dict['features'] = [deepcopy(all_plume_meta['features'][index])]
        plume_id = plume_dict['features'][0]['properties']['Plume ID']


        # rasterize that polygon
        ds = gdal.Open(os.path.join(dcid_sourcedir, f'dcid_{plume_dict["features"][0]["properties"]["DCID"]}_mf_ort.tif'))
        dat = ds.ReadAsArray().squeeze()

        rawspace_coords = rawspace_coordinate_conversion([], plume_dict['features'][0]['geometry']['coordinates'][0], ds.GetGeoTransform(), ortho=True)
        manual_mask = rasterize(shapes=[Polygon(rawspace_coords)], out_shape=(dat.shape[0],dat.shape[1])) # numpy binary mask for manual IDs

        y_locs = np.where(np.sum(manual_mask > 0, axis=1))[0]
        x_locs = np.where(np.sum(manual_mask > 0, axis=0))[0]

        dat[manual_mask < 1] = -9999
        dat = dat[y_locs[0]:y_locs[-1],x_locs[0]:x_locs[-1]]
        outtrans = list(ds.GetGeoTransform())
        outtrans[0] = outtrans[0] + x_locs[0]*outtrans[1]
        outtrans[3] = outtrans[3] + y_locs[0]*outtrans[5]

        scene_names = []
        for _s in range(len(plume_dict['features'][0]['properties']['Scene FIDs'])):
            fid =plume_dict['features'][0]['properties']['Scene FIDs'][_s]
            scene =plume_dict['features'][0]['properties']['DAAC Scene Numbers'][_s]
            orbit =plume_dict['features'][0]['properties']['Orbit']
            scene_names.append(f'EMIT_L2B_{gtype.upper()}ENH_{extra_metadata["product_version"]}_{fid[4:12]}T{fid[13:19]}_{orbit}_{scene}')

        metadata = {
            'Plume_Complex': plume_dict['features'][0]['properties']['Plume ID'],
            'Estimated_Uncertainty_ppmm': plume_dict['features'][0]['properties']['Concentration Uncertainty (ppm m)'],
            'UTC_Time_Observed': plume_dict['features'][0]['properties']['UTC Time Observed'],
            #Source_Scenes - match full conventions j
            'Source_Scenes': ','.join(scene_names),
            'Latitude of max concentration': plume_dict['features'][0]['properties']['Latitude of max concentration'],
            'Longitude of max concentration': plume_dict['features'][0]['properties']['Longitude of max concentration'],
            'Max Plume Concentration (ppm m)': plume_dict['features'][0]['properties']['Max Plume Concentration (ppm m)'],
            }
        metadata.update(extra_metadata)
        write_science_cog(dat, output_base + '.tif', outtrans, ds.GetProjection(), metadata)
        write_color_quicklook(dat, output_base + '.png', gtype=gtype)

        plume_output_file = os.path.join(output_base + '.json')
        # conger the DAAC Scene Numbers to full dac names, as above
        plume_dict['features'][0]['properties']['DAAC Scene Names'] = scene_names
        del plume_dict['features'][0]['properties']['style']
        del plume_dict['features'][0]['properties']['Data Download']
        with open(plume_output_file, 'w') as fout:
            fout.write(json.dumps(plume_dict, cls=SerialEncoder)) 


def write_color_quicklook(indat, output_file, gtype='ch4'):

    dat = indat.copy()
    mask = dat != -9999
    dat[dat < 0] = 0
    if gtype == 'ch4':
        dat = dat /1500.
        output = np.zeros((indat.shape[0],indat.shape[1],3),dtype=np.uint8)
        output[mask,:] = np.round(plt.cm.plasma(dat[mask])[...,:3] * 255).astype(np.uint8)
    elif gtype == 'co2':
        dat = dat /85000.
        output = np.zeros((indat.shape[0],indat.shape[1],3),dtype=np.uint8)
        output[mask,:] = np.round(plt.cm.viridis(dat[mask])[...,:3] * 255).astype(np.uint8)
    output[mask,:] = np.maximum(1, output[mask])


    memdriver = gdal.GetDriverByName('MEM')
    memdriver.Register()
    outDataset = memdriver.Create('',dat.shape[1],dat.shape[0],3,gdal.GDT_Byte)
    for n in range(1,4):
        outDataset.GetRasterBand(n).WriteArray(output[...,n-1])
        outDataset.GetRasterBand(n).SetNoDataValue(0)

    driver = gdal.GetDriverByName('PNG')
    driver.Register()
    dst_ds = driver.CreateCopy(output_file, outDataset, strict=0)
    del dst_ds, outDataset
    
@ray.remote
def single_scene_proc(input_file, output_file, extra_metadata,gtype='ch4'):
    ds = gdal.Open(input_file)
    dat = ds.ReadAsArray().squeeze()
    write_science_cog(dat, output_file, ds.GetGeoTransform(), ds.GetProjection(), extra_metadata)
    write_color_quicklook(dat, output_file.replace('.tif','.png'), gtype=gtype)


def get_daac_link(feature, product_version, outbasedir, delivered_plume_names, delivered_plume_ids, gtype='ch4'):
    prod_v = product_version.split('V')[-1]
    filename, complete = daac_filename(feature, delivered_plume_names, delivered_plume_ids, product_version, gtype='ch4')
    #link = f'https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2BCH4PLM.{prod_v}/EMIT_L2B_CH4PLM_{prod_v}_{fid[4:12]}T{fid[13:19]}_{cid}/EMIT_L2B_CH4PLM_{prod_v}_{fid[4:12]}T{fid[13:19]}_{cid}.tif'
    link = f'https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2B{gtype.upper()}PLM.{prod_v}/{filename}/{filename}.tif'
    if complete:
        return link

    #if len(glob.glob(os.path.join(outbasedir, fid[4:12], 'l2bch4plm', f'EMIT_L2B_CH4PLM_{prod_v}_{fid[4:12]}T{fid[13:19]}_{cid}*.json'))) > 0:
    r = requests.head(link)
    if r.status_code != 404:
        print(r.status_code, link)
        return link
    else:
        print(r.status_code, 'Coming soon')
        return 'Coming soon'

def daac_filename(feat, previous_plume_names, previous_plume_ids, prod_v, gtype='ch4'):
    if previous_plume_names is None or previous_plume_ids is None:
        return fid
    else:
        complex_id= feat['Plume ID'].split('-')[-1]
        if f'CH4_PlumeComplex-{complex_id.zfill(6)}' in previous_plume_ids:
            name = previous_plume_names[previous_plume_ids.index(f'{gtype.upper()}_PlumeComplex-{complex_id.zfill(6)}')]
            return name, True
        else:
            fid=feat['Scene FIDs'][0]
            cid= feat['Plume ID'].split('-')[-1].zfill(6)
            name = f'EMIT_L2B_{gtype.upper()}PLM_{prod_v.split("V")[-1]}_{fid[4:12]}T{fid[13:19]}_{cid}'
            return name, False

            


def main(input_args=None):
    parser = argparse.ArgumentParser(description="Delineate/colorize plume")
    parser.add_argument('--source_dir', type=str, default='methane_20230813')
    parser.add_argument('--dest_dir', type=str, default='visions_delivery')
    parser.add_argument('--manual_del_dir', type=str, default='/beegfs/scratch/brodrick/methane/ch4_plumedir_scenetest/')
    parser.add_argument('--software_version', type=str, default=None)
    parser.add_argument('--data_version', type=str, default=None)
    parser.add_argument('--visions_delivery', type=int, choices=[0,1,2],default=0)
    parser.add_argument('--n_cores', type=int, default=1)
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--type', type=str, choices=['ch4','co2'], default='ch4')
    parser.add_argument('--previous_plume_file', type=str, default=None)
    parser.add_argument('--loglevel', type=str, default='DEBUG', help='logging verbosity')    
    parser.add_argument('--logfile', type=str, default=None, help='output file to write log to')    
    args = parser.parse_args(input_args)

    logging.basicConfig(format='%(levelname)s:%(asctime)s ||| %(message)s', level=args.loglevel,
                        filename=args.logfile, datefmt='%Y-%m-%d,%H:%M:%S')

    tile_dir = os.path.join(args.dest_dir, f'{args.type}_plume_tiles')


    if args.previous_plume_file is not None:
        with open(args.previous_plume_file,'r') as f:
            delivered_plume_names = f.read().splitlines()
            delivered_plume_ids = [f'{args.type.upper()}_PlumeComplex-{x.split("_")[-1]}' for x in delivered_plume_names]
    else:
        delivered_plume_names = []
        delivered_plume_ids = []


    all_plume_meta = json.load(open(f'{args.manual_del_dir}/combined_plume_metadata.json'))
    unique_fids = np.unique([sublist for feat in all_plume_meta['features'] for sublist in feat['properties']['Scene FIDs']])
    dcids = np.array([feat['properties']['DCID'] for feat in all_plume_meta['features']])
    unique_dcids = np.unique(dcids)

    valid_plume_idx = [x for x, feat  in enumerate(all_plume_meta['features']) if feat['properties']['style']['color'] == 'white' and feat['geometry']['type'] == 'Polygon']
    valid_point_idx = [x for x, feat  in enumerate(all_plume_meta['features']) if feat['properties']['style']['color'] == 'white' and feat['geometry']['type'] == 'Point']

    plume_count = 1

    ray.init(num_cpus=args.n_cores)


    extra_metadata = {}
    if args.software_version:
        extra_metadata['software_build_version'] = args.software_version
    else:
        cmd = ["git", "symbolic-ref", "-q", "--short", "HEAD", "||", "git", "describe", "--tags", "--exact-match"]
        output = subprocess.run(" ".join(cmd), shell=True, capture_output=True)
        if output.returncode != 0:
            raise RuntimeError(output.stderr.decode("utf-8"))
        extra_metadata['software_build_version'] = output.stdout.decode("utf-8").replace("\n", "")

    if args.data_version:
        extra_metadata['product_version'] = args.data_version
    extra_metadata['keywords'] = "Imaging Spectroscopy, minerals, EMIT, dust, radiative forcing"
    extra_metadata['sensor'] = "EMIT (Earth Surface Mineral Dust Source Investigation)"
    extra_metadata['instrument'] = "EMIT"
    extra_metadata['platform'] = "ISS"
    extra_metadata['Conventions'] = "CF-1.63"
    extra_metadata['institution'] = "NASA Jet Propulsion Laboratory/California Institute of Technology"
    extra_metadata['license'] = "https://science.nasa.gov/earth-science/earth-science-data/data-information-policy/"
    extra_metadata['naming_authority'] = "LPDAAC"
    extra_metadata['date_created'] = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ")
    extra_metadata['keywords_vocabulary'] = "NASA Global Change Master Directory (GCMD) Science Keywords"
    extra_metadata['stdname_vocabulary'] = "NetCDF Climate and Forecast (CF) Metadata Convention"
    extra_metadata['creator_name'] = "Jet Propulsion Laboratory/California Institute of Technology"
    extra_metadata['creator_url'] = "https://earth.jpl.nasa.gov/emit/"
    extra_metadata['project'] = "Earth Surface Mineral Dust Source Investigation"
    extra_metadata['project_url'] = "https://earth.jpl.nasa.gov/emit/"
    extra_metadata['publisher_name'] = "NASA LPDAAC"
    extra_metadata['publisher_url'] = "https://lpdaac.usgs.gov"
    extra_metadata['publisher_email'] = "lpdaac@usgs.gov"
    extra_metadata['identifier_product_doi_authority'] = "https://doi.org"
    extra_metadata['title'] = "EMIT"
    extra_metadata['Units']= 'ppm m'

    print(delivered_plume_ids)

    if args.visions_delivery != 2:

        jobs = []
        for _feat, feat in enumerate(all_plume_meta['features']):
            if _feat not in valid_plume_idx:
                continue
            complex_id= feat['properties']['Plume ID'].split('-')[-1]
            if f'{args.type.upper()}_PlumeComplex-{complex_id.zfill(6)}' in delivered_plume_ids:
                continue
            logging.info(f'Processing plume {_feat+1}/{len(all_plume_meta["features"])}')

            if feat['geometry']['type'] == 'Polygon':
                outdir=os.path.join(args.dest_dir, feat['properties']['Scene FIDs'][0][4:12], f'l2b{args.type}plm')
                if os.path.isdir(outdir) is False:
                    subprocess.call(f'mkdir -p {outdir}',shell=True)
                output_base = os.path.join(outdir, feat['properties']['Scene FIDs'][0] + '_' + feat['properties']['Plume ID'])
                if args.overwrite or os.path.isfile(output_base + '.tif') is False:
                    jobs.append(single_plume_proc.remote(all_plume_meta, _feat, output_base, args.manual_del_dir, args.source_dir, extra_metadata, gtype=args.type))
        rreturn = [ray.get(jid) for jid in jobs]


        jobs = []
        for fid in unique_fids:
            outdir = os.path.join(args.dest_dir, fid[4:12], f'l2b{args.type}enh')
            if os.path.isdir(outdir) is False:
                subprocess.call(f'mkdir -p {outdir}',shell=True)
            of = os.path.join(outdir, fid + f'{args.type}_enh.tif')
            if args.overwrite or os.path.isfile(of) is False:
                jobs.append(single_scene_proc.remote(os.path.join(args.source_dir, fid[4:12], fid + f'_{args.type}_mf_ort'),  of, extra_metadata, gtype=args.type))
        rreturn = [ray.get(jid) for jid in jobs]



    if args.visions_delivery == 1 or args.visions_delivery == 2:

        outdir = os.path.join(args.dest_dir, f'visions_{args.type}_tiles')
        if os.path.isdir(outdir) is False:
            subprocess.call(f'mkdir -p {outdir}',shell=True)


        logging.info('Build output geojson')
        outdict = {"crs": {"properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84" }, "type": "name"},"features":[],"name":"methane_metadata","type":"FeatureCollection" }
        for nmi in valid_plume_idx:
            newfeat = all_plume_meta['features'][nmi].copy()
            pc = newfeat['properties']['Plume ID']
            complex_id= pc.split('-')[-1]

            if newfeat['geometry']['type'] == 'Polygon':
                newfeat['properties']['plume_complex_count'] = plume_count
                newfeat['properties']['Data Download'] = get_daac_link(newfeat['properties'], extra_metadata['product_version'], args.dest_dir, delivered_plume_names, delivered_plume_ids, gtype=args.type)
                outdict['features'].append(newfeat)

                for npi in valid_point_idx:
                    pointfeat = all_plume_meta['features'][npi].copy()
                    if pointfeat['properties']['Plume ID'] == newfeat['properties']['Plume ID']:
                        pointfeat['properties']['plume_complex_count'] = plume_count
                        pointfeat['properties']['Data Download'] = newfeat['properties']['Data Download']
                        pointfeat['properties']['style'] = {'color': 'red','fillOpacity':0,'maxZoom':9,'minZoom':0,'opacity':1,'radius':10,'weight':2}
                        outdict['features'].append(pointfeat)
                        break
                plume_count += 1

        with open(os.path.join(args.dest_dir, 'combined_plume_metadata.json'), 'w') as fout:
            fout.write(json.dumps(outdict, cls=SerialEncoder)) 

        if args.type == 'ch4':
            subprocess.call(f"rsync {os.path.join(args.dest_dir, 'combined_plume_metadata.json')} brodrick@137.79.132.213:/data/emit/mmgis/coverage/combined_plume_metadata.json",shell=True)
        if args.type == 'co2':
            subprocess.call(f"rsync {os.path.join(args.dest_dir, 'combined_plume_metadata.json')} brodrick@137.79.132.213:/data/emit/mmgis/coverage/combined_co2_plume_metadata.json",shell=True)

    
        logging.info('Tile output')
        for _dcid, dcid in enumerate(unique_dcids):
            logging.info(f'Tiling {_dcid + 1} / {len(unique_dcids)}')
            match_idx = np.where(dcids == dcid)[0]
            

            subfeatures = [feat for _feat, feat in enumerate(all_plume_meta['features']) if _feat in match_idx and _feat in valid_plume_idx and f'{args.type.upper()}_PlumeComplex-{(feat["properties"]["Plume ID"].split("-")[-1]).zfill(6)}' not in delivered_plume_ids]
            if len(subfeatures) > 0:
                tile_dcid(subfeatures, outdir, args.manual_del_dir, gstyle=args.type)
        subprocess.call(f"rsync -a --info=progress2 {args.dest_dir}/visions_{args.type}_tiles/ brodrick@137.79.132.213:/data/emit/mmgis/mosaics/{args.type}_plume_tiles/",shell=True)
    



if __name__ == '__main__':
    main()


