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
# Authors: Philip G Brodrick, philip.brodrick@jpl.nasa.gov

import argparse
import subprocess

import os
from utils import envi_header
from glob import glob



def main(input_args=None):
    parser = argparse.ArgumentParser(description="Robust MF")
    parser.add_argument('date', type=str,  metavar='DATE', help='path to input image')   
    parser.add_argument('output_dir', type=str,  metavar='OUTPUT', help='path to input image')   
    parser.add_argument('--co2', action='store_true', help='flag to indicate whether to run co2')    
    args = parser.parse_args(input_args)

    if args.date == 'all':
        rdn_files = glob(f'/beegfs/store/emit/ops/data/acquisitions/*/emit*/l1b/*_rdn_b0106_v01.img')
    else:
        rdn_files = glob(f'/beegfs/store/emit/ops/data/acquisitions/{args.date}/emit*/l1b/*_rdn_b0106_v01.img')
    obs_files = [x.replace('rdn','obs') for x in rdn_files]
    loc_files = [x.replace('rdn','loc') for x in rdn_files]
    glt_files = [x.replace('rdn','glt') for x in rdn_files]
    l1b_bandmask_files = [x.replace('rdn','bandmask') for x in rdn_files]
    l2a_mask_files = [x.replace('l1b','l2a').replace('rdn','mask') for x in rdn_files]

    state_files = [x.replace('l1b','l2a').replace('rdn','statesubs') for x in rdn_files]
    state_files = [x if os.path.isfile(x) else None for x in state_files]

    if os.path.isdir(os.path.join(args.output_dir, args.date)) is False:
        subprocess.call(f'mkdir {os.path.join(args.output_dir, args.date)}',shell=True)
        
    out_files = [os.path.join(args.output_dir, args.date, os.path.basename(x).split('_')[0]) for x in rdn_files]

    n=0
    for _r in range(len(rdn_files)):

      ch4_mf_kmz_file = f'{out_files[_r]}_ch4_mf_scaled_color_ort.tif'
      co2_mf_kmz_file = f'{out_files[_r]}_co2_mf_scaled_color_ort.tif'

      launch = os.path.isfile(ch4_mf_kmz_file) is False
      if os.path.isfile(ch4_mf_kmz_file) is False or (args.co2 and os.path.isfile(co2_mf_kmz_file) is False):
        cmd_str=f'sbatch -N 1 -c 64 -p standard --mem=300G --wrap="python ghg_process.py {rdn_files[_r]} {obs_files[_r]} {loc_files[_r]} {glt_files[_r]} {l1b_bandmask_files[_r]} {l2a_mask_files[_r]} {out_files[_r]}'

        if args.co2:
            cmd_str += ' --co2 --lut_file /store/shared/ghg/dataset_co2_full.hdf5'

        if state_files[_r] is not None:
            cmd_str += f' --state_subs {state_files[_r]}"'
        else:
            cmd_str += f'"'
            continue

        print(cmd_str)

        env=os.environ.copy()
        subprocess.call(cmd_str,shell=True,env=env)


if __name__ == '__main__':
    main()
