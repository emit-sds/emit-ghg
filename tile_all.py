#! /usr/bin/env python
#
#  Copyright 2022 California Institute of Technology
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
# ISOFIT: Imaging Spectrometer Optimal FITting
# Authors: Philip G. Brodrick, philip.brodrick@jpl.nasa.gov

import os
import subprocess
import argparse
import numpy as np
import glob
from spectral.io import envi


def main():

    parser = argparse.ArgumentParser(description="Run visuals workflow")
    #parser.add_argument('input_file_list', type=str)
    parser.add_argument('type',type=str, choices=['ch4','co2'])
    args = parser.parse_args()

    #loclist = np.genfromtxt(args.input_file_list,dtype=str)
    #ch4list = sorted(glob.glob('methane_20220928/*_ch4_mf_scaled_ort.tif'))
    ch4list = sorted(glob.glob('methane_20221019/*_ch4_mf_scaled_ort.tif'))
    co2list = [x.replace('ch4','co2') for x in ch4list]
    #co2list = np.genfromtxt('co2_fids.txt',dtype=str).tolist()

    run_file = f'already_run_locs_{args.type}.txt'
    if os.path.isfile(run_file):
        prerun_loc_list = np.genfromtxt(run_file,dtype=str).tolist()
    else:
        prerun_loc_list = []
    


    for _loc in range(len(ch4list)): 
    #for _loc in range(10): 
      if ch4list[_loc] not in prerun_loc_list:

        cmd_str = ''
        if args.type == 'ch4':
          if os.path.isfile(ch4list[_loc]):
            cmd_str= f'gdal2tiles.py {ch4list[_loc]} combined_ch4 -z 4-12 --srcnodata 0 -r antialias --processes=40'
            #cmd_str= f'python gdal2tilesemit.py {ch4list[_loc]} combined_ch4 -z 4-12 --srcnodata -9999 -r antialias'
            #cmd_str= f'python gdal2customtiles.py {ch4list[_loc]} combined_ch4 -z 4-12 --srcnodata -9999 --dem'
            #cmd_str= f'python gdal2customtiles.py {ch4list[_loc]} combined_ch4 -z 4-12 --srcnodata -9999 --dem'
        elif args.type == 'co2':
          if os.path.isfile(co2list[_loc]):
            cmd_str= f'gdal2tiles.py {co2list[_loc]} combined_co2 -z 4-12 --srcnodata 0 -r antialias --processes=40'

        if cmd_str != '':

            cmd_str += f'&& echo {ch4list[_loc]} >> {run_file}'
            cmd_str = f'sbatch -N 1 -c 40 --mem=180G -p standard --wrap="{cmd_str}"'
            subprocess.call(cmd_str,shell=True)

        

if __name__ == "__main__":
    main()
