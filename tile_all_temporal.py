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

import os
import subprocess
import argparse
import glob


def main():

    parser = argparse.ArgumentParser(description="Run visuals workflow")
    parser.add_argument('dates', type=str, nargs='+')
    parser.add_argument('--sourcedir', type=str, default='methane_20230813')
    parser.add_argument('--type', type=str, default='ch4', choices=['co2','ch4'])
    args = parser.parse_args()

    path = os.environ['PATH']
    path = path.replace('\Library\\bin;',':')
    os.environ['PATH'] = path

    tiled_basedir = os.path.join(args.sourcedir, f'{args.type}_mosaic_temporal_static')
    sens_basedir = os.path.join(args.sourcedir, f'{args.type}_sens_mosaic_temporal_static')
    uncert_basedir = os.path.join(args.sourcedir, f'{args.type}_uncert_mosaic_temporal_static')
    for dname in [tiled_basedir, sens_basedir, uncert_basedir]:
        if os.path.isdir(dname) is False:
            subprocess.call(f'mkdir -p {dname}', shell=True)


    if args.dates[0] == 'all':
        dates = [os.path.basename(x) for x in glob.glob('/beegfs/store/emit/ops/data/acquisitions/202*')]
    else:
        dates = args.dates

    for date in dates:
        
        if args.type == 'ch4':
            static_file_list =  os.path.join(tiled_basedir, f'{date}_methane_static.txt')
            sens_file_list =  os.path.join(sens_basedir, f'{date}_methane_sens_static.txt')
            uncert_file_list =  os.path.join(uncert_basedir, f'{date}_methane_uncert_static.txt')
        elif args.type == 'co2':
            static_file_list =  os.path.join(tiled_basedir, f'{date}_co2_static.txt')
            sens_file_list =  os.path.join(sens_basedir, f'{date}_co2_sens_static.txt')
            uncert_file_list =  os.path.join(uncert_basedir, f'{date}_co2_uncert_static.txt')

        subprocess.call(f'find {os.path.join(args.sourcedir, date)} -name "emit{date}*_{args.type}_mf_scaled_color_ort.tif" > {static_file_list}',shell=True)
        subprocess.call(f'find {os.path.join(args.sourcedir, date)} -name "emit{date}*_{args.type}_sens_scaled_color_ort.tif" > {sens_file_list}',shell=True)
        subprocess.call(f'find {os.path.join(args.sourcedir, date)} -name "emit{date}*_{args.type}_uncert_scaled_color_ort.tif" > {uncert_file_list}',shell=True)

        od_date = f'{date[:4]}-{date[4:6]}-{date[6:8]}T00_00_01Z-to-{date[:4]}-{date[4:6]}-{date[6:8]}T23_59_59Z'

        
        subprocess.call(f'sbatch -N 1 -c 40 --mem=180G --job-name {args.type}_tile_{date} --wrap="python daily_tiler.py {static_file_list} {os.path.join(tiled_basedir, od_date)}"',shell=True)
        subprocess.call(f'sbatch -N 1 -c 40 --mem=180G --job-name {args.type}_tile_{date} --wrap="python daily_tiler.py {sens_file_list}   {os.path.join(sens_basedir, od_date)}"',shell=True)
        subprocess.call(f'sbatch -N 1 -c 40 --mem=180G --job-name {args.type}_tile_{date} --wrap="python daily_tiler.py {uncert_file_list} {os.path.join(uncert_basedir, od_date)}"',shell=True)

        

if __name__ == "__main__":
    main()

