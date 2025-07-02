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
import json
import numpy as np



def main(input_args=None):
    parser = argparse.ArgumentParser(description="Robust MF")
    parser.add_argument('output_dir', type=str,  metavar='OUTPUT', help='path to input image')   
    parser.add_argument('--co2', action='store_true', help='flag to indicate whether to run co2')    
    parser.add_argument('--methane_metadata', type=str, default='visions_delivery/combined_plume_metadata.json', help='file_with_idd_methane_plumes')    
    args = parser.parse_args(input_args)

    plumedict = json.load(open(args.methane_metadata,'r'))

    #fids = [x['properties']['Scene FIDs'] for x in plumedict['features']]
    fids = np.unique([sublist for feat in plumedict['features'] for sublist in feat['properties']['Scene FIDs']])

    rdn_files = []
    for fid in fids:
        glist = glob(f'/store/emit/ops/data/acquisitions/{fid[4:12]}/{fid}/l1b/*_rdn_b0106_v01.img')
        if len(glist) > 0:
            rdn_files.append(glist[0])
        else:
            rdn_files.append(None)
    for _fid in range(len(rdn_files)-1,-1,-1):
        if rdn_files[_fid] is None:
            rdn_files.pop(_fid)
            fids.pop(_fid)
    fids = np.unique(fids)

    obs_files = [x.replace('rdn','obs') for x in rdn_files]
    loc_files = [x.replace('rdn','loc') for x in rdn_files]
    glt_files = [x.replace('rdn','glt') for x in rdn_files]
    l1b_bandmask_files = [x.replace('rdn','bandmask') for x in rdn_files]
    l2a_mask_files = [x.replace('l1b','l2a').replace('rdn','mask') for x in rdn_files]

    state_files = [x.replace('l1b','l2a').replace('rdn','statesubs') for x in rdn_files]
    state_files = [x if os.path.isfile(x) else None for x in state_files]

    for fid in fids:
        date = fid[4:12]
        if os.path.isdir(os.path.join(args.output_dir, date)) is False:
            subprocess.call(f'mkdir {os.path.join(args.output_dir, date)}',shell=True)
        
    out_files = [os.path.join(args.output_dir, os.path.basename(x).split('_')[0][4:12], os.path.basename(x).split('_')[0]) for x in rdn_files]

    n=0
    for _r in range(len(rdn_files)):

      ch4_mf_kmz_file = f'{out_files[_r]}_ch4_mf_color.kmz'
      co2_mf_kmz_file = f'{out_files[_r]}_co2_mf_color.kmz'

      launch = os.path.isfile(ch4_mf_kmz_file) is False
      if os.path.isfile(ch4_mf_kmz_file) is False or (args.co2 and os.path.isfile(co2_mf_kmz_file) is False):
        cmd_str=f'sbatch -N 1 -c 64 -p standard --mem=300G --wrap="python ghg_process.py {rdn_files[_r]} {obs_files[_r]} {loc_files[_r]} {glt_files[_r]} {l1b_bandmask_files[_r]} {l2a_mask_files[_r]} {out_files[_r]}'

        if args.co2:
            cmd_str += ' --co2 --lut_file /store/shared/ghg/dataset_co2_full.hdf5'

        if state_files[_r] is not None:
            cmd_str += f' --state_subs {state_files[_r]}"'
        else:
            cmd_str += f'"'

        print(cmd_str)

        env=os.environ.copy()
        subprocess.call(cmd_str,shell=True,env=env)


if __name__ == '__main__':
    main()

