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
import datetime

import os
import json
import glob



def main(input_args=None):
    parser = argparse.ArgumentParser(description="Delineate/colorize plume")
    parser.add_argument('annotations', type=str)
    parser.add_argument('--only_after', type=str)
    args = parser.parse_args(input_args)

    annotations = json.load(open(args.annotations))

    source_dir = 'methane_20221121'
    dest_dir = 'public_mmgis_masked'
    tile_dir = os.path.join(dest_dir, 'ch4_plume_tiles')

    after_dt = None
    if args.only_after is None:
        after_dt = datetime.datetime.strptime('2022-05-01T21:31:16', "%Y-%m-%dT%H:%M:%S")
    else:
        after_dt = datetime.datetime.strptime(args.only_after, "%Y-%m-%dT%H:%M:%S")


    subprocess.call(f'mkdir {dest_dir}',shell=True)
    for entry in annotations:
        if after_dt is not None:
           dt = datetime.datetime.strptime(entry['updated_at'][:19],"%Y-%m-%dT%H:%M:%S")
           #dt = datetime.datetime.strptime(os.path.basename(entry['data']['image_mf']).split('_')[0], "emit%Y%m%dt%H%M%S")
           if dt <= after_dt:
               print(f'skipping dt {dt}, ref: {after_dt}')
               continue


        #lfid = os.path.basename(entry['image']).split('-')[1].split('_')[0]
        lfid = os.path.basename(entry['data']['image_mf']).split('_')[0]

        #local_mask = None
        #for ann in entry['annotations']:
        #    for res in ann['result']:
        #        if res['value']['format'] == 'rle':
        #            if local_mask is None:
        #                local_mask = np.zeros((res['original_height'],res['original_width']), dtype=bool)
        #            local_mask
                    

        # Oldschool
        aid = entry['annotations'][0]['id']
        try:
            mask_files = glob.glob(os.path.join(os.path.dirname(args.annotations), f'*-annotation-{aid}-*.npy'))
        except:
            print(f'nothing at annotation: {aid}')
            continue
        if len(mask_files) == 0:
            print(f'nothing at annotation: {aid}')
            continue


        output_file = os.path.join(dest_dir, f'{lfid}_ch4_plumes.tif')
        mask_output = os.path.join(dest_dir, f'{lfid}_ch4_mask.tif')
        glt_file = sorted(glob.glob(f'/beegfs/store/emit/ops/data/acquisitions/{lfid[4:12]}/{lfid.split("_")[0]}/l1b/*_glt_b0106_v01.img'))[-1]
        runargs = [os.path.join(source_dir,f'{lfid}_ch4_mf'), glt_file, output_file]
        for mask_file in mask_files:
            runargs.append(mask_file)
        runargs.extend(['-output_mask', mask_output])
        runargs.extend(['-daac_demo_dir', 'demo_daac_data'])

        date=lfid[4:]
        time=lfid.split('t')[-1]
        od_date = f'{date[:4]}-{date[4:6]}-{date[6:8]}T{time[:2]}_{time[2:4]}_{time[4:]}Z-to-{date[:4]}-{date[4:6]}-{date[6:8]}T{time[:2]}_{time[2:4]}_{str(int(time[4:6])+1):02}Z'

        cmd_str = f'sbatch -N 1 -c 40 -p standard --mem=180G --wrap="python masked_plume_delineator.py {" ".join(runargs)} && gdal2tiles.py -z 2-12 --srcnodata 0 --processes=40 -r antialias {output_file} {tile_dir}/{od_date} -x"'
        #cmd_str = f'sbatch -N 1 -c 2 --mem=30G -p standard,debug --wrap="python masked_plume_delineator.py {" ".join(runargs)}"'
        #subprocess.call(cmd_str, shell=True)
        print(cmd_str)






if __name__ == '__main__':
    main()
