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
import subprocess
import pandas as pd
import numpy as np
from skimage import draw
import os
from osgeo import gdal
import matplotlib.pyplot as plt
from datetime import datetime
import json

from glob import glob


def write_output_file(source_ds, output_img, output_file):
    driver = gdal.GetDriverByName('GTiff')
    driver.Register()
    output_tiff = output_file.replace('.png','.tif')
    outDataset = driver.Create(output_tiff ,source_ds.RasterXSize,source_ds.RasterYSize,3,gdal.GDT_Byte,options = ['COMPRESS=LZW'])
    #outDataset.SetProjection(source_ds.GetProjection())
    #outDataset.SetGeoTransform(source_ds.GetGeoTransform())
    for n in range(1,4):
        if len(output_img.shape) == 2:
            outDataset.GetRasterBand(n).WriteArray(output_img)
        else:
            outDataset.GetRasterBand(n).WriteArray(output_img[...,n-1])
        outDataset.GetRasterBand(n).SetNoDataValue(0)
    del outDataset
    subprocess.call(f'gdal_translate {output_tiff} {output_file} -of PNG -co ZLEVEL=9',shell=True) 
    subprocess.call(f'rm {output_tiff}',shell=True)



def main(input_args=None):
    path = os.environ['PATH']
    path = path.replace('\Library\\bin;',':')
    os.environ['PATH'] = path

    parser = argparse.ArgumentParser(description="Prepare for annotations")
    parser.add_argument('previous_annotation', type=str)
    parser.add_argument('--source_dir', type=str, default='methane_20221121')
    parser.add_argument('--out_dir_disk', type=str, default='plumes/annotate_prep_sbs/')
    parser.add_argument('--out_dir_ann', type=str, default='/data/local-files/?d=annotate_prep_sbs/')
    args = parser.parse_args(input_args)

    if args.previous_annotation == 'none':
        annotations = None
    else:
        annotations = json.load(open(args.previous_annotation))


    # Download / load latest 
    subprocess.call('rclone copy la_gdrive:/EMIT_Priority_Scenes_20220816.xlsx plumes/',shell=True)
    df = pd.read_excel('plumes/EMIT_Priority_Scenes_20220816.xlsx',sheet_name='Phil_Scratch', header=0)
    
    
    mmgis = np.array(df['Include in paper and MMGIS (1=yes, 0=no)'])
    subset = mmgis == 1
    fid = np.array(df['FID (new)'])[subset]
    lat = np.array(df['Plume Latitude (deg)'])[subset]
    lon = np.array(df['Plume Longitude (deg)'])[subset]

    order = np.argsort(fid)
    fid = fid[order]
    lat = lat[order]
    lon = lon[order]

    un_fid = np.unique(fid)
    #prev_fid = [os.path.basename(x['data']['image']).split('_')[0].split('-')[1] for x in annotations]
    prev_fid = [os.path.basename(x['data']['image_mf']).split('_')[0] for x in annotations]
    print(prev_fid)

    print(un_fid)
    
    output_annotations = {}
    for _l, lfid in enumerate(prev_fid):
        if lfid not in un_fid.tolist():
            print(f'Could not find {lfid} in file list - but previous annotation exists.  Please check.')
            #exit()
        output_annotations[lfid] = annotations[_l]
        if 'annotations' in output_annotations[lfid].keys():
            for _a in range(len(output_annotations[lfid]['annotations'])):
                for _r in range(len(output_annotations[lfid]['annotations'][_a]['result'])):
                    output_annotations[lfid]['annotations'][_a]['result'][_r]['to_name'] = 'image_mf'

    
    bounds = [0,1000]
    
    for _lfid, lfid in enumerate(un_fid):
    
        subset = np.where(fid == lfid)[0]

        mf_files = [os.path.join(args.source_dir, f'{lfid}_ch4_mf'), os.path.join(args.source_dir, f'{lfid}_ch4_mf_refined')]
        out_files = [os.path.join(args.out_dir_disk, f'{lfid}_for_annotation.png'),os.path.join(args.out_dir_disk, f'{lfid}_for_annotation_refined.png')]

        if lfid not in output_annotations.keys():
            output_annotations[lfid] = {'data': {}}
            
        output_annotations[lfid]['data']['image_mf'] = out_files[0].replace(args.out_dir_disk,args.out_dir_ann)
        output_annotations[lfid]['data']['image_mf_refined'] = out_files[1].replace(args.out_dir_disk,args.out_dir_ann)
        output_annotations[lfid]['file_upload'] = os.path.basename(out_files[0])
        if 'image' in output_annotations[lfid]['data'].keys():
            del output_annotations[lfid]['data']['image']
        output_annotations[lfid]['id'] = _lfid+1


        for _m in range(len(mf_files)):
            mf_file = mf_files[_m]
            out_file = out_files[_m]
            glt_file = glob(f'/beegfs/store/emit/ops/data/acquisitions/{lfid[4:12]}/{lfid}/l1b/*_glt_b0106_v01.img')[0]
            glt_ds = gdal.Open(glt_file)
            trans = glt_ds.GetGeoTransform()



            if os.path.isfile(out_file):
                continue
    
            ds = gdal.Open(mf_file)
            dat = ds.ReadAsArray().astype(np.float32)
            dat[dat == ds.GetRasterBand(1).GetNoDataValue()] = np.nan
    
            dat -= bounds[0]
            dat /= (bounds[1] - bounds[0])
            isnan = np.isnan(dat)
            dat[dat <= 0] = 0.01
            dat[isnan] = 0
            dat = plt.cm.plasma(dat)[...,:3]
            dat = np.round(dat * 255).astype(np.uint8)
            dat[isnan,:] = 0
    
            for ind in subset:
                glt_ypx = int(round((lat[ind] - trans[3])/ trans[5]))
                glt_xpx = int(round((lon[ind] - trans[0])/ trans[1]))
    
                if glt_xpx < 0 or glt_ypx < 0 or glt_xpx > glt_ds.RasterXSize or glt_ypx > glt_ds.RasterYSize:
                    glt = [0,0]
                else:
                    glt = np.squeeze(glt_ds.ReadAsArray(glt_xpx, glt_ypx, 1, 1))
    
                rr, cc = draw.circle_perimeter(glt[1], glt[0], radius=10, shape=dat.shape)
                dat[rr, cc, :] = np.array([1,255,1])
    
            write_output_file(ds, dat, out_file)



    
    output_annotations = list(output_annotations.values())
    with open(os.path.join(args.out_dir_disk, f'input_annotations_{datetime.now().strftime("%Y%m%dt%H%M%S")}.json'), 'w') as fout:
        fout.write(json.dumps(output_annotations)) 


if __name__ == '__main__':
    main()
